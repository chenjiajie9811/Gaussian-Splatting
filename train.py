import os
import cv2
import sys
import uuid
import torch
import numpy as np
from tqdm import tqdm
from random import randint

from renderer.gaussian_renderer import render

from scene.test_scene import TestScene
from scene.gaussian_model import GaussianModel

from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.system_utils import load_config

from scene.datasets import TestReplicaDataset, Replica, TUM_RGBD

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def save_rendered_image(image, gt_image, iteration):
    print ("Saving Rendered image")
    im = image.clone().cpu().detach().numpy().transpose(1, 2, 0)*255
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = np.nan_to_num(im, nan=0, posinf=255, neginf=0)
    im = np.clip(im, 0, 255).astype(np.uint8)

    gt_im = gt_image.clone().cpu().detach().numpy().transpose(1, 2, 0)*255
    gt_im = cv2.cvtColor(gt_im, cv2.COLOR_RGB2BGR)

    cv2.imwrite("./output/point_cloud/iteration_{}/rendered.png".format(iteration), im)
    cv2.imwrite("./output/point_cloud/iteration_{}/gt.png".format(iteration), gt_im)

def training(cfg, scene : TestScene, saving_iterations):
    cfg_training = cfg['training']
    cfg_model = cfg['model']
    
    first_iter = 0
    scene.gaussians.training_setup(cfg_training)

    bg_color = [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    iterations = cfg_training['iterations']
    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, iterations):        

        iter_start.record()

        scene.gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, cfg['pipeline'], background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg_training['lambda_dssim']) * Ll1 + cfg_training['lambda_dssim'] * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # print ("gaussian shape: ", gaussians.get_xyz.shape)
                # print ("gaussian mean: ", gaussians.get_xyz.mean(dim=0))
                save_rendered_image(image, gt_image, iteration)
                scene.save(iteration)

            # Densification
            if iteration < cfg_training['densify_until_iter']:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg_training['densify_from_iter'] and iteration % cfg_training['densification_interval'] == 0:
                    size_threshold = 20 if iteration > cfg_training['opacity_reset_interval'] else None
                    gaussians.densify_and_prune(cfg_training['densify_grad_threshold'], 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % cfg_training['opacity_reset_interval'] == 0 or (cfg_model['white_background'] and iteration == cfg_training['densify_from_iter']):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


    


if __name__ == "__main__":
    cfg = load_config('./config.yaml')

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset = Replica(cfg, None)
    #dataset = TestReplicaDataset(cfg['data']['root'], cfg['data']['idx0'], cfg['data']['idx1'])
    gaussians = GaussianModel(cfg['model']['sh_degree'])
    scene = TestScene(cfg, gaussians, dataset)

    training(cfg, scene, [i for i in range(50, 30000, 500)])



