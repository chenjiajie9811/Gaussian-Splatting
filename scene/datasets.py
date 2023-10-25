import os
import torch

import cv2
import glob
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.camera_utils import as_intrinsics_matrix

class TestReplicaDataset(Dataset):
    def __init__(self, root, idx1=0, idx2=100, device='cuda:0'):
        self.root = root
        self.color_paths = sorted(
            glob.glob(f'{self.root}/results/frame*.jpg'))[idx1 : idx2]
        self.depth_paths = sorted(
            glob.glob(f'{self.root}/results/depth*.png'))[idx1 : idx2]
        

        self.n_img = len(self.color_paths)
        
        self.png_depth_scale = 1. #6553.5
        
        traj_path = os.path.join(self.root, "traj.txt")

        with open(traj_path, "r") as f:
            lines = f.readlines()

        self.poses = []
        for line in lines:
            self.poses.append(self.convert_line_to_pose(line))
        self.poses = self.poses[idx1 : idx2]

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = self.read_color(color_path)
        depth_data = self.read_depth(depth_path)

        pose = self.poses[index]
        
        ret = {
            "c2w" : pose, 
            "color" : color_data, 
            "depth" : depth_data,
            "color_path" : color_path,
            "depth_path" : depth_path
        }
        return ret
    
    def read_color(self, color_path):
        color_data = cv2.imread(color_path)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        return torch.from_numpy(color_data)
    
    def read_depth(self, depth_path):
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        # H, W = depth_data.shape
        return torch.from_numpy(depth_data)
    
    def convert_line_to_pose(self, line):
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()

        return c2w

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(cfg, args, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        
        self.input_folder = cfg['data']['input_folder']
        # if args.input_folder is None:
        #     self.input_folder = cfg['data']['input_folder']
        # else:
        #     self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

    def __len__(self):
        return self.n_img

    @staticmethod
    def set_edge_pixels_to_zero(depth_data, crop_edge):
        mask = torch.ones_like(depth_data)
        mask[:crop_edge, :] = 0
        mask[-crop_edge:, :] = 0
        mask[:, :crop_edge] = 0
        mask[:, -crop_edge:] = 0

        depth_data = depth_data * mask
        return depth_data

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        ret = {
            "idx" : index,
            "c2w" : pose, 
            "color" : color_data, 
            "depth" : depth_data,
            "color_path" : color_path,
            "depth_path" : depth_path
        }

        return ret
        # return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)


class Replica(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # The codebase assumes that the camera coordinate system is X left to right,
            # Y down to up and Z in the negative viewing direction. Most datasets assume
            # X left to right, Y up to down and Z in the positive viewing direction.
            # Therefore, we need to rotate the camera coordinate system.
            # Multiplication of R_x (rotation aroun X-axis 180 degrees) from the right.
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, device)
        self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            # The codebase assumes that the camera coordinate system is X left to right,
            # Y down to up and Z in the negative viewing direction. Most datasets assume
            # X left to right, Y up to down and Z in the positive viewing direction.
            # Therefore, we need to rotate the camera coordinate system.
            # Multiplication of R_x (rotation aroun X-axis 180 degrees) from the right.
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            # The codebase assumes that the camera coordinate system is X left to right,
            # Y down to up and Z in the negative viewing direction. Most datasets assume
            # X left to right, Y up to down and Z in the positive viewing direction.
            # Therefore, we need to rotate the camera coordinate system.
            # Multiplication of R_x (rotation aroun X-axis 180 degrees) from the right.
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD
}
