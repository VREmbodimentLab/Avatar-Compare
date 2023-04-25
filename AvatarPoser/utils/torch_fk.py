import torch.nn as nn
import numpy as np
import torch
from human_body_prior.tools.tgm_conversion import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_angle_axis

class ForwardKinematics(nn.Module):
    def __init__(self, offsets, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        super(ForwardKinematics, self).__init__()
        self.register_buffer('offsets', offsets)
        if norm_idx is not None:
            self.offsets = self.offsets / np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints
        
    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: torch tensor of shape (N, n_joints*3*3)
        Returns:
            The 3D joint positions as a tensor of shape (N, n_joints, 3)
        """
        assert joint_angles.shape[-1] == self.n_joints * 9
        angles = joint_angles.view(-1, self.n_joints, 3, 3).to('cuda')
        n_frames = angles.size(0)
        device = angles.device
        if self.left_mult:
            offsets = self.offsets.view(1, 1, self.n_joints, 3)
        else:
            offsets = self.offsets.view(1, self.n_joints, 3, 1).to(device)

        if self.no_root:
            angles[:, 0] = torch.eye(3).to(device)

        assert self.parents[0] == -1
        positions = {0: torch.zeros([n_frames, 3]).to(device)}
        rotations = {0: angles[:, 0].to(device)}

        for j in range(1, self.n_joints):
            prev_rotation = rotations[self.parents[j]].to(device)
            prev_position = positions[self.parents[j]].to(device)
            # this is a regular joint
            if self.left_mult:
                position = torch.squeeze(torch.matmul(offsets[:, :, j], prev_rotation)) + prev_position
                rotation = torch.matmul(angles[:, j], prev_rotation)
            else:
                position = torch.squeeze(torch.matmul(prev_rotation, offsets[:, j])) + prev_position
                rotation =  torch.matmul(prev_rotation, angles[:, j])
            positions[j] = position
            rotations[j] = rotation
                
        return torch.cat([positions[j].view(n_frames, 1, 3) for j in range(self.n_joints)], 1)
    
    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, n_joints*3).
        """
        angles_rot = Rotation.from_rotvec(joint_angles).as_matrix()
        return self.fk(torch.reshape(angles_rot, [-1, self.n_joints * 9]))
    
    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, n_joints*3*3).
        """
        return self.fk(joint_angles)
    
    def from_quat(self, joint_angles):
        raise NotImplementedError()

    def from_sparse(self, joint_angles_sparse, rep="rotmat", return_sparse=True):
        raise NotImplementedError()
    


SMPLH_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPLH_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPLH_SKELETON = np.array([
    [ 0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19],
    [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
]).T

class SMPLHForwardKinematics(ForwardKinematics):
    """
    Forward Kinematics for the skeleton defined by SMPL-H.
    """
    def __init__(self):
        # this are the offsets stored under `J` in the SMPL model pickle file
        offsets = torch.tensor([[-1.79505953e-03, -2.23333446e-01,  2.82191255e-02],
                           [ 6.77246757e-02, -3.14739671e-01,  2.14037877e-02],
                           [-6.94655406e-02, -3.13855126e-01,  2.38993038e-02],
                           [-4.32792313e-03, -1.14370215e-01,  1.52281192e-03],
                           [ 1.02001221e-01, -6.89938274e-01,  1.69079858e-02],
                           [-1.07755594e-01, -6.96424140e-01,  1.50492738e-02],
                           [ 1.15910534e-03,  2.08102144e-02,  2.61528404e-03],
                           [ 8.84055199e-02, -1.08789863e+00, -2.67853442e-02],
                           [-9.19818258e-02, -1.09483879e+00, -2.72625243e-02],
                           [ 2.61610388e-03,  7.37324481e-02,  2.80398521e-02],
                           [ 1.14763659e-01, -1.14368952e+00,  9.25030544e-02],
                           [-1.17353574e-01, -1.14298274e+00,  9.60854266e-02],
                           [-1.62284535e-04,  2.87602804e-01, -1.48171829e-02],
                           [ 8.14608431e-02,  1.95481750e-01, -6.04975478e-03],
                           [-7.91430834e-02,  1.92565283e-01, -1.05754332e-02],
                           [ 4.98955543e-03,  3.52572414e-01,  3.65317875e-02],
                           [ 1.72437770e-01,  2.25950646e-01, -1.49179062e-02],
                           [-1.75155461e-01,  2.25116450e-01, -1.97185045e-02],
                           [ 4.32050017e-01,  2.13178586e-01, -4.23743412e-02],
                           [-4.28897421e-01,  2.11787231e-01, -4.11194829e-02],
                           [ 6.81283645e-01,  2.22164620e-01, -4.35452575e-02],
                           [-6.84195501e-01,  2.19559526e-01, -4.66786778e-02],])

        # need to convert them to compatible offsets
        smplh_offsets = torch.zeros([22, 3])
        smplh_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPLH_PARENTS[1:]):
            smplh_offsets[idx+1] = offsets[idx + 1] - offsets[pid]

        # normalize so that right thigh has length 1
        super(SMPLHForwardKinematics, self).__init__(smplh_offsets, SMPLH_PARENTS, norm_idx=4,
                                                    left_mult=False, major_joints=SMPLH_MAJOR_JOINTS)
        
        
class Rotation():
    """
    Class to give a scipy.spatial.transform-like interface
    for converting between rotational formalisms in PyTorch.
    Acts on trailing axes and maintains leading tensor shape.
    """
    def __init__(self, tensor, shape, formalism):
        self.tensor, self.shape = tensor, shape
        assert formalism in ['rotvec', 'matrix', 'quat_scalar_first', 'quat_scalar_last']
        self.formalism = formalism

    @staticmethod
    def from_rotvec(x):
        return Rotation(x.view(-1, 3), x.size(), 'rotvec')
        
    @staticmethod
    def from_matrix(x):
        return Rotation(x.view(-1, 3, 3), x.size(), 'matrix')
    
    def as_rotvec(self):
        if self.formalism == 'matrix':
            s = self.shape
            if s[-1]%9 == 0:
                new_shape = s[:-1] + (s[-1]//3,)
            elif s[-1]%3 == 0:
                new_shape = s[:-2] + (3,)
            else:
                raise NotImplementedError()
            rotvec = rotation_matrix_to_angle_axis(F.pad(self.tensor, [0,1])) # why is this padded??
            return rotvec.reshape(*new_shape)
        elif self.formalism == 'rotvec':
            return self.tensor.reshape(self.shape)
        elif 'quat' in self.formalism:
            if self.formalism == 'quat_scalar_last':
                perm = torch.tensor([3, 0, 1, 2], dtype=torch.long).to(self.tensor.device)
                self.tensor = self.tensor[:, perm]
            s = self.shape
            rotvec = quaternion_to_angle_axis(quaternion)
            return rotvec.reshape(*s[:-1], 3*s[-1]//4)
        else:
            raise NotImplementedError()

    def as_matrix(self):
        if self.formalism == 'matrix':
            return self.tensor.reshape(self.shape)
        elif self.formalism == 'rotvec':
            s = self.shape
            matrot = angle_axis_to_rotation_matrix(self.tensor)[:, :3, :3].contiguous()
            return matrot.view(*s[:-1], s[-1]*3)
        elif 'quat' in self.formalism:
            self = self.from_rotvec(self.as_rotvec())
            return self.as_matrix()
        else:
            raise NotImplementedError()

    def from_euler(self):
        raise NotImplementedError()
        
    @staticmethod
    def from_quat(x, scalar_last=True):
        if scalar_last:
            return Rotation(x.view(-1, 4), x.size(), 'quat_scalar_last')
        else:
            return Rotation(x.view(-1, 4), x.size(), 'quat_scalar_first')

    def from_mrp(self):
        raise NotImplementedError()

    def as_euler(self, degrees=False):
        if degrees:
            raise NotImplementedError("Degrees as output not supported.")
        if self.formalism == 'rotvec':
            self = Rotation.from_matrix(self.as_matrix())
        elif self.formalism != 'matrix':
            raise NotImplementedError()
        rs = self.tensor
        n_samples = rs.size(0)
        
        # initialize to zeros
        e1 = torch.zeros([n_samples]).to(rs.device)
        e2 = torch.zeros([n_samples]).to(rs.device)
        e3 = torch.zeros([n_samples]).to(rs.device)
        
        # find indices where we need to treat special cases
        is_one = rs[:, 0, 2] == 1
        is_minus_one = rs[:, 0, 2] == -1
        is_special = torch.logical_or(is_one, is_minus_one)
        
        e1[is_special] = torch.atan2(rs[is_special, 0, 1], rs[is_special, 0, 2])
        e2[is_minus_one] = np.pi/2
        e2[is_one] = -np.pi/2
        
        # normal cases
        is_normal = torch.logical_not(torch.logical_or(is_one, is_minus_one))
        # clip inputs to arcsin
        in_ = torch.clamp(rs[is_normal, 0, 2], -1, 1)
        e2[is_normal] = -torch.arcsin(in_)
        e2_cos = torch.cos(e2[is_normal])
        e1[is_normal] = torch.atan2(rs[is_normal, 1, 2]/e2_cos,
                                    rs[is_normal, 2, 2]/e2_cos)
        e3[is_normal] = torch.atan2(rs[is_normal, 0, 1]/e2_cos,
                                    rs[is_normal, 0, 0]/e2_cos)

        eul = torch.stack([e1, e2, e3], axis=-1)
        #eul = np.reshape(eul, np.concatenate([orig_shape, eul.shape[1:]]))
        s = self.shape
        eul = eul.reshape(*s[:-1], s[-1]//3)
        return eul

    @staticmethod
    def _to_scalar_last(x):
        perm = torch.tensor([1, 2, 3, 0], dtype=torch.long).to(x.device)
        return x[:, perm]
        
    def as_quat(self):
        if 'quat' in self.formalism:
            if self.formalism == 'quat_scalar_first':
                self.tensor = self._to_scalar_last(self.tensor)
            return self.tensor.reshape(self.shape)
        elif self.formalism == 'rotvec':
            self = Rotation.from_matrix(self.as_matrix())
        assert self.formalism == 'matrix'
        s = self.shape
        if s[-1]%9 == 0:
            new_shape = s[:-1] + (4*s[-1]//9,)
        elif s[-1] == 3:
            new_shape = s[:-2] + (4,)
        quat = rotation_matrix_to_quaternion(F.pad(self.tensor, [0,1]))
        quat = self._to_scalar_last(quat)
        return quat.reshape(*new_shape)

    def as_mrp(self):
        raise NotImplementedError()