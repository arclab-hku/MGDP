from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.envs import LeggedRobot
from .legged_robot_config_baseline import LeggedRobotBaseCfg, LeggedRobotBaseCfgPPO
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
import copy
import math
from termcolor import cprint

class CameraMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_handle_trans(self, width, height, env_idx, trans, rot, hfov=None):
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True

        if hfov is not None:
            camera_props.horizontal_fov = hfov
        camera_handle = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*trans)
        local_transform.r = gymapi.Quat.from_euler_zyx(*rot)
        return camera_handle, local_transform
    
    def make_handle_trans_points(self, width, height, env_idx, trans, rot, hfov=None):
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True

        if hfov is not None:
            camera_props.horizontal_fov = hfov
        camera_handle = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*trans)
        local_transform.r = gymapi.Quat.from_euler_zyx(*rot)
        return camera_handle, local_transform

class Legged_camera(LeggedRobot):
    cfg: LeggedRobotBaseCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.use_lidar = cfg.camera.use_lidar
        self.use_camera = cfg.camera.use_camera
        self.image_nums = cfg.camera.image_nums
        self.resized = cfg.camera.resized
        

        if self.cfg.camera.camera_type=='isaac' and self.use_camera:
            self.im = []
            self.camera_handles = []
            self.camera_res = cfg.camera.camera_res
            self.horizontal_fov = cfg.camera.horizontal_fov
            for i in range(self.num_envs):
                cam1, trans1 = self.make_handle_trans(self.camera_res[0], self.camera_res[1], i, cfg.camera.trans,  cfg.camera.rot1)
                self.camera_handles.append(cam1)
                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "base"
                )
                self.gym.attach_camera_to_body(
                    cam1,  # camera_handle,
                    self.envs[i],
                    body_handle,
                    trans1,
                    gymapi.FOLLOW_TRANSFORM,
                )
                self.cam_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    gymapi.IMAGE_DEPTH,
                )
                self.im.append(gymtorch.wrap_tensor(self.cam_tensor))


    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.camera.near_clip) / (
                    self.cfg.camera.far_clip - self.cfg.camera.near_clip) - 0.5
        return depth_image


    def process_depth_image(self):
        # cropping on an image
        if self.use_camera:
            image = torch.stack(self.im, dim=0)
            image += self.cfg.camera.depth_nosie * 2 * (torch.rand(1) - 0.5)[0]
            image = torch.clamp(-image, self.cfg.camera.near_clip, self.cfg.camera.far_clip)

        if self.cfg.camera.hight_point_type == "resize":
            resize_transform = transforms.Resize((self.resized[1], self.resized[0]), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image = resize_transform(image)
        elif self.cfg.camera.hight_point_type == "nearest":
            image = F.interpolate(image.unsqueeze(0), size=self.resized, mode='nearest')
            image = image.squeeze(0)
        elif self.cfg.camera.hight_point_type == "bilinear":
            image = F.interpolate(image.unsqueeze(0), size=self.resized, mode='bilinear', align_corners=False)
            image = image.squeeze(0)
        elif self.cfg.camera.hight_point_type == "bilinear_align_corners":
            image = F.interpolate(image.unsqueeze(0), size=self.resized, mode='bilinear', align_corners=True)
            image = image.squeeze(0)
        elif self.cfg.camera.hight_point_type == "bicubic":
            image = F.interpolate(image.unsqueeze(0), size=self.resized, mode='bicubic', align_corners=False)
            image = image.squeeze(0)
        elif self.cfg.camera.hight_point_type == "min_pooling":
            target_size = (self.resized[0], self.resized[1])
            kernel_size = (image.shape[1] // target_size[0], image.shape[2]// target_size[1])
            stride = kernel_size
            image = F.max_pool2d(-image, kernel_size=kernel_size, stride=stride)
            image = -image.squeeze(0)
        elif self.cfg.camera.hight_point_type == "average_pooling":
            target_size = (self.resized[0], self.resized[1])
            kernel_size = (image.shape[1] // target_size[0], image.shape[2]// target_size[1])
            stride = kernel_size
            image = F.max_pool2d(image, kernel_size=kernel_size, stride=stride)
            image = image.squeeze(0)
        else:
            pass
        image = image / (self.cfg.camera.far_clip - self.cfg.camera.near_clip)
        return image


