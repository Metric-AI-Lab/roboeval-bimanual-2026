"""Robot-only MuJoCo kinematics; no environment or task/world-state access.

The private model holds only Panda body constants, hinges and two EE sites.
All qpos values come from the controller's own command history. No dynamics,
contacts, rendering, task objects or privileged simulator inputs are used.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.book_robot_model import ROBOT_MODEL


def _values(values):
    return ' '.join(repr(float(v)) for v in values)


class RobotKinematics:
    def __init__(self, base_position):
        self.base_position = np.asarray(base_position, float).copy()
        root = ET.Element('mujoco', model='policy_robot_kinematics')
        ET.SubElement(root, 'compiler', angle='radian', fusestatic='false', balanceinertia='true')
        world = ET.SubElement(root, 'worldbody')
        elements, joint = [], 0
        for index, body in enumerate(ROBOT_MODEL['bodies']):
            parent = world if body['parent'] < 0 else elements[body['parent']]
            element = ET.SubElement(parent, 'body', name=f'b{index}', pos=_values(body['pos']), quat=_values(body['quat']))
            elements.append(element)
            if body['mass'] > 0:
                ET.SubElement(element, 'inertial', mass=repr(body['mass']), pos=_values(body['ipos']),
                              quat=_values(body['iquat']), diaginertia=_values(body['inertia']))
            for hinge in body['joints']:
                ET.SubElement(element, 'joint', name=f'q{joint}', type='hinge', axis=_values(hinge['axis']),
                              pos=_values(hinge['pos']), limited='false', ref=repr(hinge['qpos0']))
                joint += 1
        for side, site in enumerate(ROBOT_MODEL['sites']):
            ET.SubElement(elements[site['body']], 'site', name=f'ee{side}', pos=_values(site['pos']), quat=_values(site['quat']))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        # Preserve compiled public constants exactly; recompilation can
        # renormalize quaternions by one last bit, significant near IK rank loss.
        for index, body in enumerate(ROBOT_MODEL['bodies']):
            b = self.model.body(f'b{index}').id
            for field in ('pos', 'quat', 'ipos', 'iquat', 'mass', 'inertia'):
                getattr(self.model, 'body_'+field)[b] = body[field]
        for side, site in enumerate(ROBOT_MODEL['sites']):
            s = self.model.site(f'ee{side}').id
            self.model.site_pos[s], self.model.site_quat[s] = site['pos'], site['quat']
        self.root_body = self.model.body('b0').id
        self.site_ids = [self.model.site(f'ee{i}').id for i in (0, 1)]
        self.data = mujoco.MjData(self.model)
        self.initial_qctrl = np.asarray(ROBOT_MODEL['initial_qctrl'], float)
        self._quat = np.empty(4)

    def _set(self, joints, world=False):
        self.model.body_pos[self.root_body] = self.base_position if world else 0.
        self.data.qpos[:] = np.asarray(joints, float)
        mujoco.mj_fwdPosition(self.model, self.data)

    def poses(self, joints, world=False):
        self._set(joints, world=world)
        return [(self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3,3).copy()) for s in self.site_ids]

    def command_pose(self, joints):
        # Reproduce the real robot's float32 q -> world FK -> pelvis-relative
        # float32 pose convention, including the base translation's rounding.
        self._set(np.asarray(joints, np.float32), world=True)
        out = np.empty(12, np.float32)
        for side, site in enumerate(self.site_ids):
            out[6*side:6*side+3] = self.data.site_xpos[site]-self.base_position
            mujoco.mju_mat2Quat(self._quat, self.data.site_xmat[site])
            out[6*side+3:6*side+6] = R.from_quat(self._quat, scalar_first=True).as_rotvec()
        return out.astype(float)

    def solve(self, joints, side, target_position, target_rotation):
        """Same numerical update/layout as dm_control qpos_from_site_pose."""
        self._set(joints)
        start = 7*side
        original = np.asarray(joints, float)[start:start+7].copy()
        if self._solve_current(side, target_position, target_rotation):
            return self.data.qpos[start:start+7].copy()
        return original

    def update(self, joints, action, max_joint_delta):
        """Solve both arms on one IK state, then apply fallback and joint clamp.

        The reference solver leaves the UNCLAMPED left solution in its state
        while solving the right arm, even when left IK eventually falls back.
        Clamping between solves changes numerical Jacobians for bimanual moves.
        """
        original = np.asarray(joints, float).copy()
        ee = self.command_pose(original).astype(np.float32)
        targets = []
        for side in (0, 1):
            j = 6*side
            p = (ee[j:j+3]+action[j:j+3]).astype(float)
            rv = (R.from_rotvec(action[j+3:j+6].astype(float))*
                  R.from_rotvec(ee[j+3:j+6].astype(float))).as_rotvec().astype(np.float32)
            targets.append((p,R.from_rotvec(rv.astype(float))))
        self._set(original)
        success = [self._solve_current(side,*targets[side]) for side in (0,1)]
        solution = self.data.qpos.copy()
        for side in (0,1):
            if not success[side]:
                solution[7*side:7*side+7] = original[7*side:7*side+7]
        return original+np.clip(solution-original,-max_joint_delta,max_joint_delta)

    def _solve_current(self, side, target_position, target_rotation):
        site = self.site_ids[side]
        start = 7*side
        indices = list(range(start, start+7))
        mujoco.mj_fwdPosition(self.model,self.data)
        target_quat = target_rotation.as_quat(scalar_first=True)
        site_quat, neg_quat, err_quat = np.empty(4), np.empty(4), np.empty(4)
        jac = np.empty((6, self.model.nv))
        err = np.empty(6)
        update = np.zeros(self.model.nv)
        for _ in range(40):
            err[:3] = target_position-self.data.site_xpos[site]
            mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[site])
            mujoco.mju_negQuat(neg_quat, site_quat)
            mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
            mujoco.mju_quat2Vel(err[3:], err_quat, 1.)
            norm = np.linalg.norm(err[:3])+np.linalg.norm(err[3:])
            if norm < 1e-5:
                return True
            mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], site)
            selected = jac[:, indices]  # Keep the reference's Fortran layout.
            hess = selected.T.dot(selected)
            rhs = selected.T.dot(err)
            if norm > .1:
                hess += np.eye(7)*.03
                dq = np.linalg.solve(hess, rhs)
            else:
                dq = np.linalg.lstsq(hess, rhs, rcond=-1)[0]
            update_norm = np.linalg.norm(dq)
            if update_norm == 0 or norm/update_norm > 20.:
                break
            if update_norm > 2.:
                dq *= 2./update_norm
            update[indices] = dq
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.)
            mujoco.mj_fwdPosition(self.model, self.data)
        return False
