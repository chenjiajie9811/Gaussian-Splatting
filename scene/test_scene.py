import os
import cv2
import torch
import numpy as np

from PIL import Image
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.datasets import TestReplicaDataset, Replica

from utils.general_utils import PILtoTorch, getNerfppNorm
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud, geom_transform_points

def pc_transform(pc, transform):
    ones = np.ones((pc.shape[0], 1), dtype=pc.dtype)
    points_hom = np.concatenate([pc, ones], axis=1)
    points_out = np.dot(transform, points_hom.T).T
    #points_out = np.dot(c2w, points_hom.T).T

    denom = points_out[:, 3:] + 0.0000001
    points_out = (points_out[:, :3] / denom)

    # R = transform[:3, :3].numpy()
    # T = transform[:3, 3].numpy()
    # points_out = R.dot(pc.T) + T.reshape(-1, 1)
    return points_out

class TestScene:
    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, dataset):
        self.gaussians = gaussians
        self.model_path = "./output/"

        print(dataset[0]["color"].shape)
        self.w = dataset[0]["color"].shape[1]
        self.h = dataset[0]["color"].shape[0]
        self.fx = cfg['cam']['fx']
        self.fy = cfg['cam']['fy']
        self.cx = cfg['cam']['cx']
        self.cy = cfg['cam']['cy']

        self.train_cameras = []
        cnt = 0
        for idx, data in enumerate(dataset):
            self.train_cameras.append(self.construct_cam(idx, data["c2w"], data["color_path"]))
            cnt += 1
            if cnt > 10:
                break

        nerf_normalization = getNerfppNorm(self.train_cameras)
        self.cameras_extent = nerf_normalization["radius"]
        
        init_pc = self.generate_init_pc(dataset[0]["depth"], dataset[0]["color"], 512, dataset[0]["c2w"])
        
        self.draw_reprojection_points(init_pc, dataset[0]["color"], dataset[10]["color"], dataset[0]["c2w"], dataset[10]["c2w"])
        
        self.gaussians.create_from_pcd(init_pc, self.cameras_extent)

        self.save(0)

    def draw_reprojection_points(self, init_pc : BasicPointCloud, color1_, color2_, c2w1, c2w2):
        init_pc_cam1 = pc_transform(init_pc.points, c2w1)
        init_pc_cam2 = pc_transform(init_pc.points, c2w2)

        u1 = init_pc_cam1[:, 0] * self.fx / init_pc_cam1[:, 2] + self.cx
        v1 = init_pc_cam1[:, 1] * self.fy / init_pc_cam1[:, 2] + self.cy

        u1 = u1[~np.isnan(u1)]
        v1 = v1[~np.isnan(v1)]

        u2 = init_pc_cam2[:, 0] * self.fx / init_pc_cam2[:, 2] + self.cx
        v2 = init_pc_cam2[:, 1] * self.fy / init_pc_cam2[:, 2] + self.cy

        u2 = u2[~np.isnan(u2)]
        v2 = v2[~np.isnan(v2)]


        color1 = color1_.clone().numpy() * 255
        color2 = color2_.clone().numpy() * 255

        for i in range(u1.shape[0]):
            
            cv2.circle(color1, (int(u1[i]), int(v1[i])), 1, (0, 0, 255), 1)
            cv2.circle(color2, (int(u2[i]), int(v2[i])), 1, (0, 0, 255), 1)

        cv2.imwrite('./repro1.png', color1)
        cv2.imwrite('./repro2.png', color2)

        print("Write reprojection images")



    def generate_init_pc(self, depth, color, n, c2w):
        # Randomly sample points
        i, j = torch.meshgrid(torch.linspace(
                0, self.w-1, self.w), torch.linspace(0, self.h-1, self.h), indexing='ij')
        i = i.t().reshape(-1)
        j = j.t().reshape(-1)
        indices = torch.randint(i.shape[0], (n,))
        indices = indices.clamp(0, i.shape[0])
        i = i[indices]
        j = j[indices]

        color_img = color.clone().numpy() * 255
        for idx in range(i.shape[0]):
            cv2.circle(color_img, (int(i[idx]), int(j[idx])), 1, (0, 0, 255), 1)
        cv2.imwrite('./sampled0.png', color_img)


        depth = depth.reshape(-1)
        color = color.reshape(-1, 3)
        depth_sampled = depth[indices]
        color_sampled = color[indices]

        pts3D_x = depth_sampled * (i - self.cx) / self.fx
        pts3D_y = depth_sampled * (j - self.cy) / self.fy

        pts3D = np.hstack([pts3D_x.reshape(-1, 1), pts3D_y.reshape(-1, 1), depth_sampled.reshape(-1, 1)])
        normals = np.zeros((n, 3))

        pts3D_world = pc_transform(pts3D, np.linalg.inv(c2w))
        pc = BasicPointCloud(pts3D_world, color_sampled.numpy(), normals)
        #pc = BasicPointCloud(np.vstack([pts3D_world, pts3D]), np.vstack([color_sampled.numpy(), color_sampled.numpy()]), np.vstack([normals, normals]))
        return pc

    
    def construct_cam(self, id, c2w, image_path):
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        FovX = focal2fov(self.fx, self.w)
        FovY = focal2fov(self.fy, self.h)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image_torch = PILtoTorch(image, (self.w, self.h))[:3, ...]

        camera = Camera(colmap_id=id, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image_torch, gt_alpha_mask=None, image_name=image_name, uid=id, data_device="cuda:0")
        
        return camera

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras