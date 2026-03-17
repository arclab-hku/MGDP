from legged_gym.envs import LeggedRobot
from legged_gym.envs import Legged_terrains
from legged_gym.envs import Legged_rewards
from legged_gym.envs import Legged_camera, CameraMixin
from .random_dog_config_stage1 import RandomCfgStage1, RandomCfgPPOStage1
import cv2, os
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import random
from termcolor import cprint
import torch.nn as nn
import torch.nn.functional as F


class Randomdog(CameraMixin, Legged_terrains, Legged_camera, Legged_rewards, LeggedRobot):
    cfg: RandomCfgStage1

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.global_counter = 0
        self.use_world_model = self.cfg.camera.world_model
        self.use_memory = self.cfg.camera.use_memory
        self.use_map_decoder = self.cfg.camera.use_map_decoder
        self.load_world_model_policy = self.cfg.camera.load_world_model_policy
        self.noise_gaussian = self.cfg.camera.noise_gaussian
        self.noise_dropout = self.cfg.camera.noise_dropout
        self.normalize = self.cfg.camera.normalize

        self.terrain_adaptive_reward = getattr(self.cfg.rewards, 'terrain_adaptive_reward', True)

        if self.use_world_model:
            from .utils.net_work import ImageEncoder, Memory, ImageDecoder, MapEncoder, MapDecoder

            EncoderCfg = getattr(self.cfg, 'encoder_ppo_ref', None) or RandomCfgPPOStage1
            CNNModule_info = EncoderCfg.Encoder.CNNModule_info
            self.cnn_encoder = ImageEncoder(
                input_channels=CNNModule_info["input_channels"],
                hidden_channels=CNNModule_info["hidden_channels"],
                output_channels=CNNModule_info["output_channels"],
                pool=CNNModule_info["pool"]
            ).to(self.device)

            decoder_hidden_ch = CNNModule_info["hidden_channels"][::-1]
            self.cnn_decoder = ImageDecoder(
                input_dims=CNNModule_info["output_channels"],
                hidden_channels=decoder_hidden_ch,
                pool=CNNModule_info["pool"]
            ).to(self.device)

            if self.use_map_decoder:
                MapModule_info = EncoderCfg.Encoder.MapModule_info
                self.map_encoder = MapEncoder(
                    input_channels=MapModule_info["input_channels"],
                    hidden_channels=MapModule_info["hidden_channels"],
                    output_channels=MapModule_info["output_channels"],
                    pool=MapModule_info["pool"]
                ).to(self.device)

                self.map_decoder = MapDecoder(
                    input_dims=MapModule_info["output_channels"],
                    hidden_channels=MapModule_info["hidden_channels"][::-1],
                    pool=MapModule_info["pool"]
                ).to(self.device)

            if self.use_memory:
                GRUModule_info = EncoderCfg.Encoder.GRUModule_info
                self.hidden_states = None
                self.memory_lstm = Memory(GRUModule_info["input_dims"],
                                          GRUModule_info["rnn_type"],
                                          GRUModule_info["rnn_num_layers"],
                                          GRUModule_info["rnn_hidden_dims"]).to(self.device)

            self.temperature = 0.1
            self.info_nce_loss = nn.CrossEntropyLoss()
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()

            if self.load_world_model_policy:
                from legged_gym import LEGGED_GYM_ROOT_DIR
                wm_file = 'wm_best.pt' if (getattr(self.cfg.camera, 'update_wm', True) is False) else 'wm_best.pt'
                world_model_path = self.cfg.camera.load_world_model_policy_file.format(
                    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR) + "/stage1_nn/" + wm_file
                cprint(f"load world model path: {world_model_path}", 'yellow', attrs=['bold'])

                world_model_dict = torch.load(world_model_path, map_location=self.device)
                self.cnn_encoder.load_state_dict(world_model_dict['image_encoder'])
                self.cnn_decoder.load_state_dict(world_model_dict['image_decoder'])
                if self.use_map_decoder:
                    if 'map_encoder' in world_model_dict and 'map_decoder' in world_model_dict:
                        self.map_encoder.load_state_dict(world_model_dict['map_encoder'])
                        self.map_decoder.load_state_dict(world_model_dict['map_decoder'])
                        cprint("World model: map_encoder/map_decoder loaded from checkpoint", 'green')
                    else:
                        cprint("World model: map_encoder/map_decoder NOT in checkpoint -> prediction may be constant (random init)", 'red', attrs=['bold'])
                if self.use_memory and 'memory_encoder' in world_model_dict:
                    self.memory_lstm.load_state_dict(world_model_dict['memory_encoder'])

                self.set_wm_mode(
                    update_wm=self.cfg.camera.update_wm,
                    world_model_dict=world_model_dict
                )
            else:
                self.set_wm_mode(
                    update_wm=self.cfg.camera.update_wm,
                )



        if self.use_camera:
            if self.image_nums is None:
                self.image_buf = torch.zeros(
                    self.num_envs,
                    1,
                    self.resized[0],  # rows = height
                    self.resized[1],  # cols = width
                    device=self.device,
                    dtype=torch.float,
                )

                self.image_clean_buf = torch.zeros(
                    self.num_envs,
                    1,
                    self.resized[0],  # rows = height
                    self.resized[1],  # cols = width
                    device=self.device,
                    dtype=torch.float,
                )
            else:
                self.image_buf = torch.zeros(
                    self.num_envs,
                    self.image_nums,
                    self.resized[0],  # rows = height
                    self.resized[1],  # cols = width
                    device=self.device,
                    dtype=torch.float,
                )
            self.image_clean_buf = torch.zeros(
                self.num_envs,
                self.image_nums,
                self.resized[0],  # rows = height
                self.resized[1],  # cols = width
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.image_buf = None

        if (self.use_camera or self.use_lidar) and self.cfg.camera.camera_type == 'warp':
            from warp_sensor import WarpManager, Config as WarpConfig
            from .utils.sensor_config import Camera, Lidar, ImageProcess
            cfg = WarpConfig()
            if self.cfg.camera.use_camera == True:
                cfg.camera.pattern = Camera.pattern
                cfg.camera.offset = Camera.offset
                cfg.camera.process = Camera.process
                print('camera_config', Camera())
            if self.cfg.camera.use_lidar == True:
                cfg.lidar.pattern = Lidar().pattern
                print('lidar_config', Lidar().pattern)
            else:
                cfg.lidar = None

            print()
            self.terrain.vertices -= np.array([self.terrain.cfg.border_size, self.terrain.cfg.border_size, 0])
            self.warp_manager = WarpManager(self.num_envs, self, cfg=cfg, device=self.device)



    def set_wm_mode(self, update_wm, world_model_dict=None):
        """
        Set train/eval and gradient flags for world model. Restore optimizer state from world_model_dict if given.
        """
        self._update_wm = update_wm
        self.cnn_encoder.train(update_wm)
        self.cnn_decoder.train(update_wm)

        if self.use_map_decoder:
            self.map_encoder.train(update_wm)
            self.map_decoder.train(update_wm)

        if self.use_memory:
            self.memory_lstm.train(update_wm)

        for param in self.cnn_encoder.parameters():
            param.requires_grad = update_wm
        for param in self.cnn_decoder.parameters():
            param.requires_grad = update_wm

        if self.use_map_decoder:
            for param in self.map_encoder.parameters():
                param.requires_grad = update_wm
            for param in self.map_decoder.parameters():
                param.requires_grad = update_wm

        if self.use_memory:
            for param in self.memory_lstm.parameters():
                param.requires_grad = update_wm


        params_encoder = list(self.cnn_encoder.parameters())
        if self.use_map_decoder:
            params_encoder += list(self.map_encoder.parameters())
        if self.use_memory:
            params_encoder += list(self.memory_lstm.parameters())
        params_decoder = list(self.cnn_decoder.parameters())
        if self.use_map_decoder:
            params_decoder += list(self.map_decoder.parameters())

        wm_lr = getattr(self.cfg.camera, 'wm_lr', 1e-4)
        wm_decoder_lr_scale = getattr(self.cfg.camera, 'wm_decoder_lr_scale', 1.0)
        wm_weight_decay = getattr(self.cfg.camera, 'wm_weight_decay', 0.0)
        params = [
            {'params': params_encoder, 'lr': wm_lr, 'weight_decay': 0.0},
            {'params': params_decoder, 'lr': wm_lr * wm_decoder_lr_scale, 'weight_decay': wm_weight_decay},
        ]

        self.wm_optimizer = torch.optim.Adam(params)
        if update_wm:
            if world_model_dict is not None and 'wm_optimizer' in world_model_dict:
                self.wm_optimizer.load_state_dict(world_model_dict['wm_optimizer'])
                cprint("World model set to TRAIN (from resume) mode", 'green')
            else:
                cprint("World model set to TRAIN (from scratch) mode", 'green')
        else:
            if world_model_dict is not None and 'wm_optimizer' in world_model_dict:
                self.wm_optimizer.load_state_dict(world_model_dict['wm_optimizer'])
            cprint("World model set to EVAL mode (frozen)", 'green')

    def get_camera_sensor(self):
        if self.cfg.camera.camera_type == 'isaac':
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            cur_images = self.process_depth_image()
        elif self.cfg.camera.camera_type == 'warp':
            self.warp_manager.warp_update_frame()
            if self.cfg.camera.use_camera == True:
                cur_images = self.warp_manager['camera'].data.squeeze()

            if self.cfg.camera.use_lidar == True:
                self.cur_lidar = self.warp_manager['lidar'].data.squeeze()
                # print('cur_images',  self.cur_lidar.shape)

            noisy_cur_images = cur_images.clone()
            if self.noise_gaussian is not None:
                noisy_cur_images.add_(torch.randn_like(noisy_cur_images) * self.noise_gaussian)
            if self.noise_dropout is not None:
                mask = torch.rand_like(noisy_cur_images) > self.noise_dropout
                noisy_cur_images.mul_(mask)


            if self.normalize:
                near = float(self.cfg.camera.near_clip)
                far = float(self.cfg.camera.far_clip)
                depth_range = far - near
                if depth_range < 1e-6:
                    depth_range = 1e-6
                cur_images = ((cur_images.clamp(near, far) - near) / depth_range).clamp(0.0, 1.0)
                noisy_cur_images = ((noisy_cur_images.clamp(near, far) - near) / depth_range).clamp(0.0, 1.0)

            if self.image_nums:
                self.image_clean_buf = torch.cat([
                    self.image_clean_buf[:, 1:, :, :],
                    cur_images.unsqueeze(1)
                ], dim=1)

                self.image_buf = torch.cat([
                    self.image_buf[:, 1:, :, :],
                    noisy_cur_images.unsqueeze(1)
                ], dim=1)

            else:
                self.image_clean_buf = cur_images
                self.image_buf = noisy_cur_images

        if self.image_nums == 1 or self.image_nums == None:
            pass
        else:
            at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(at_reset_env_ids) > 0:
                self.image_buf[at_reset_env_ids, 0, :, :] = noisy_cur_images[at_reset_env_ids, :, :]
                self.image_buf[at_reset_env_ids, 1, :, :] = noisy_cur_images[at_reset_env_ids, :, :]

                self.image_clean_buf[at_reset_env_ids, 0, :, :] = cur_images[at_reset_env_ids, :, :]
                self.image_clean_buf[at_reset_env_ids, 1, :, :] = cur_images[at_reset_env_ids, :, :]

        if self.cfg.camera.camera_type == 'isaac':
            self.gym.end_access_image_tensors(self.sim)

        if self.use_world_model:
            self.visual_l1_loss = 0
            self.visual_mse_loss = 0

            self.height_map_bce_loss = 0
            self.contrastive_loss = 0

            visual_input = self.image_buf.to(self.device)
            batch_size = visual_input.size(0)

            if not getattr(self, '_update_wm', self.cfg.camera.update_wm):
                with torch.no_grad():
                    visual_pre, obs_height_pre, obs_height_input, visual_token, self.visual_mse_loss, self.visual_l1_loss, \
                        self.height_mse_loss, self.height_l1_loss, self.contrastive_loss, \
                        self.height_visual_mse_loss = self.compute_world_model_outputs(
                        visual_input, batch_size, update_wm=False
                    )

                    self.total_wm_loss = (1.0 * self.visual_mse_loss + 0.0 * self.visual_l1_loss +
                                          1.0 * self.height_mse_loss + 0.0 * self.height_l1_loss +
                                          0.0 * self.height_visual_mse_loss +
                                          0.0 * self.contrastive_loss)
            else:
                visual_input = self.image_buf.to(self.device).clone().detach().requires_grad_(True)
                (visual_pre, obs_height_pre, obs_height_input, visual_token, self.visual_mse_loss,
                 self.visual_l1_loss, \
                    self.height_mse_loss, self.height_l1_loss, self.contrastive_loss, \
                    self.height_visual_mse_loss) = self.compute_world_model_outputs(
                    visual_input, batch_size, update_wm=True
                )

                w_v_mse = getattr(self.cfg.camera, 'wm_visual_mse_weight', 0.7)
                w_v_l1 = getattr(self.cfg.camera, 'wm_visual_l1_weight', 0.3)
                w_h_mse = getattr(self.cfg.camera, 'wm_height_mse_weight', 1.0)
                w_h_l1 = getattr(self.cfg.camera, 'wm_height_l1_weight', 0.2)
                w_hv = getattr(self.cfg.camera, 'wm_height_visual_mse_weight', 0.7)
                w_cont = getattr(self.cfg.camera, 'wm_contrastive_weight', 0.3)
                self.total_wm_loss = (w_v_mse * self.visual_mse_loss + w_v_l1 * self.visual_l1_loss +
                                      w_h_mse * self.height_mse_loss + w_h_l1 * self.height_l1_loss +
                                      w_hv * self.height_visual_mse_loss + w_cont * self.contrastive_loss)
                self.total_wm_loss.requires_grad_(True)

                self.wm_optimizer.zero_grad()
                self.total_wm_loss.backward()
                wm_grad_clip = getattr(self.cfg.camera, 'wm_grad_clip', None)
                if wm_grad_clip is not None and wm_grad_clip > 0:
                    all_wm_params = []
                    for g in self.wm_optimizer.param_groups:
                        all_wm_params += g['params']
                    torch.nn.utils.clip_grad_norm_(all_wm_params, max_norm=float(wm_grad_clip))
                self.wm_optimizer.step()
            if self.cfg.camera.render_compare_pre_map and self.use_map_decoder and obs_height_pre is not None:
                pred_height = obs_height_pre[0, 0].detach().cpu().numpy()
                true_height = obs_height_input[0, 0].detach().cpu().numpy()
                if getattr(self.cfg.camera, 'debug_height_stats', False) and self.global_counter % 200 == 0:
                    pred_t = obs_height_pre[0, 0].detach()
                    true_t = obs_height_input[0, 0].detach()
                    cprint(f"[WM height] step={self.global_counter} pred: min={pred_t.min().item():.4f} max={pred_t.max().item():.4f} mean={pred_t.mean().item():.4f} std={pred_t.std().item():.4f} | true: min={true_t.min().item():.4f} max={true_t.max().item():.4f} mean={true_t.mean().item():.4f} std={true_t.std().item():.4f}", 'cyan')
                if pred_height.shape != true_height.shape:
                    true_height = cv2.resize(
                        true_height,
                        (pred_height.shape[1], pred_height.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                comparison_img = self.draw_sparse_heatmap(pred_height, true_height, cell_size=60)
                cv2.imshow("Sparse Height Comparison", comparison_img)
                cv2.waitKey(1)

            if self.cfg.camera.render_compare_pre_vis:
                scale_factor = 20
                vis_env_id = min(1, visual_pre.size(0) - 1)
                img_np = visual_pre[vis_env_id, 0].detach().cpu().numpy()
                new_size = (int(img_np.shape[1] * scale_factor), int(img_np.shape[0] * scale_factor))
                img_pre_scaled = cv2.resize(img_np, new_size, interpolation=cv2.INTER_LINEAR)

                img_noise_np = self.image_buf[vis_env_id, 0].detach().cpu().numpy()
                img_noise_scaled = cv2.resize(img_noise_np, new_size, interpolation=cv2.INTER_LINEAR)

                img_clean_np = self.image_clean_buf[vis_env_id, 0].detach().cpu().numpy()
                img_clean_scaled = cv2.resize(img_clean_np, new_size, interpolation=cv2.INTER_LINEAR)

                combined = np.hstack((img_noise_scaled, img_pre_scaled, img_clean_scaled))
                cv2.imshow("Noise and Clean Image", combined)
                cv2.waitKey(1)

            if self.use_memory:
                visual_token, self.hidden_states = self.memory_lstm(visual_token, self.hidden_states)

            if self.load_world_model_policy:
                self.obs_dict['image_buf'] = visual_token.to(self.device).detach()
            else:
                self.obs_dict['image_buf'] = visual_token.to(self.device).detach()
        else:
            self.obs_dict['image_buf'] = self.image_buf.to(self.device)

    def compute_world_model_outputs(self, visual_input, batch_size, update_wm=False):
        """Forward and loss for depth (visual) and height map. Returns visual_pre, height_pre, inputs, token, losses."""
        visual_input = torch.nan_to_num(visual_input, nan=0.0, posinf=1.0, neginf=0.0)
        visual_token = self.cnn_encoder(visual_input)
        visual_pre = self.cnn_decoder(
            visual_token,
            encoder_skips=self.cnn_encoder.skip_connections,
            target_size=(self.image_clean_buf.size(2), self.image_clean_buf.size(3))
        )
        visual_pre = visual_pre.clamp(0.0, 1.0)
        target_clean = self.image_clean_buf.clamp(0.0, 1.0)
        visual_mse_loss = self.mse_loss(visual_pre, target_clean)
        visual_l1_loss = self.compute_edge_loss(visual_pre, target_clean)

        obs_height_pre = None
        obs_height_input = None
        height_mse_loss = height_l1_loss = contrastive_loss = height_visual_mse_loss = 0.0
        if self.use_map_decoder:
            n_rows = getattr(self.cfg.terrain, 'num_point_x', 17)
            n_cols = getattr(self.cfg.terrain, 'num_point_y', 11)
            num_points = n_rows * n_cols
            n_size = int((self.heights.shape[1] / max(1, num_points)) ** 0.5)
            obs_height_input = (self.heights / self.obs_scales.height_measurements).view(
                batch_size, 1, n_size * n_rows, n_size * n_cols
            )
            obs_height_input = torch.clamp(obs_height_input, -1.0, 1.0)
            obs_height_input = torch.nan_to_num(obs_height_input, nan=0.0, posinf=1.0, neginf=-1.0)
            if update_wm:
                obs_height_input = obs_height_input.clone().detach().requires_grad_(True)

            height_token = self.map_encoder(obs_height_input)
            obs_height_pre = self.map_decoder(
                height_token,
                encoder_skips=self.map_encoder.skip_connections,
                target_size=(n_rows, n_cols)
            )

            height_mse_loss = self.mse_loss(obs_height_pre, obs_height_input)
            height_l1_loss = self.compute_edge_loss(obs_height_pre, obs_height_input)

            height_token_norm = F.normalize(height_token, p=2, dim=1, eps=1e-8)
            visual_token_32 = visual_token[:, 0:32]
            visual_token_norm = F.normalize(visual_token_32, p=2, dim=1, eps=1e-8)
            if batch_size >= 2:
                sim_matrix = torch.matmul(height_token_norm, visual_token_norm.T) / self.temperature
                labels = torch.arange(batch_size, device=self.device)
                contrastive_loss = self.info_nce_loss(sim_matrix, labels)
                height_visual_mse_loss = self.mse_loss(height_token_norm, visual_token_norm)
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device)
                height_visual_mse_loss = torch.tensor(0.0, device=self.device)


        return visual_pre, obs_height_pre, obs_height_input,visual_token, visual_mse_loss, visual_l1_loss, \
            height_mse_loss, height_l1_loss, contrastive_loss, \
            height_visual_mse_loss

    def compute_edge_loss(self, pred, target):
        """Sobel edge L1 loss for depth/height map reconstruction (B,C,H,W)."""
        import torch
        import torch.nn.functional as F
        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=torch.float32,
            device=pred.device
        )
        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=torch.float32,
            device=pred.device
        )

        C = pred.shape[1]
        sobel_x = sobel_x.repeat(C, 1, 1, 1)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)

        pred_edge_x = F.conv2d(
            pred,
            weight=sobel_x,
            padding=1,
            groups=C
        )
        pred_edge_y = F.conv2d(pred, weight=sobel_y, padding=1, groups=C)

        target_edge_x = F.conv2d(target, weight=sobel_x, padding=1, groups=C)
        target_edge_y = F.conv2d(target, weight=sobel_y, padding=1, groups=C)

        edge_loss = (
                torch.abs(pred_edge_x - target_edge_x).mean() +
                torch.abs(pred_edge_y - target_edge_y).mean()
        )

        return edge_loss

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)

        if self.cfg.terrain.mesh_type == 'gap_parkour':
            # self.reset_buf[self.env_class == 14] *= False
            self.reset_buf[self.env_class == 17] *= False
            self.reset_buf[self.env_class == 14] *= False

            self.reset_buf_1 = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices_narrow, :], dim=-1) > 1., dim=1)
            self.reset_buf_1[self.env_class != 17] *= False
            self.reset_buf |= self.reset_buf_1

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        if self.cfg.terrain.mesh_type == 'gap_parkour':
            base_id = self.env_class > 2

            base_cutoff = (self.root_states[:, 2] < -0.1)
            base_cutoff &= base_id
            self.reset_buf |= base_cutoff

            min_foot_height, _ = torch.min(self.foot_height, dim=1)
            foot_cutoff = (min_foot_height < -0.1)
            foot_cutoff &= base_id
            self.reset_buf |= foot_cutoff

        if self.cfg.terrain.mesh_type == 'mix':
            # base_id = (self.env_class == 7).nonzero(as_tuple=True)[0]

            base_id = self.env_class == 6
            height_cutoff = self.root_states[:, 2] < -0.2
            height_cutoff &= base_id
            self.reset_buf |= height_cutoff

            # min_foot_height, _ = torch.min(self.foot_height, dim=1)
            # foot_cutoff = (min_foot_height < -0.1)
            # foot_cutoff &= base_id
            # self.reset_buf |= foot_cutoff

    def compute_observations(self):
        """ Computes observations
        """
        super().compute_observations()
        if self.use_camera:
            if self.global_counter % self.cfg.camera.update_interval == 0:
                self.get_camera_sensor()
            elif self.use_world_model:
                self.visual_mse_loss = torch.tensor(0.0, device=self.device)
                self.visual_l1_loss = torch.tensor(0.0, device=self.device)
                self.height_mse_loss = torch.tensor(0.0, device=self.device)
                self.height_l1_loss = torch.tensor(0.0, device=self.device)
                self.contrastive_loss = torch.tensor(0.0, device=self.device)
                self.height_visual_mse_loss = torch.tensor(0.0, device=self.device)
                self.total_wm_loss = torch.tensor(0.0, device=self.device)

    def reset(self):
        super().reset()
        if self.use_memory:
            self.memory_lstm.hidden_states = None
        if self.use_camera is True:
            if self.use_world_model is True:
                x_cnn = self.image_buf.to(self.device)
                visual_token = self.cnn_encoder(x_cnn)
                if self.use_memory:
                    visual_token, self.hidden_states = self.memory_lstm(visual_token, None)
                self.obs_dict['image_buf'] = visual_token.to(self.device)
            else:
                self.obs_dict['image_buf'] = self.image_buf.to(self.device)
        else:
            pass
        return self.obs_dict

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if self.use_camera is True and self.use_memory is True:
            self.memory_lstm.reset(self.reset_buf)

    def step(self, actions):
        super().step(actions)
        self.global_counter += 1

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def _post_physics_step_callback(self):
        real_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        self.mask_pace = (real_vel < self.cfg.rewards.gait_threshold[0])
        self.mask_trot = (real_vel <= self.cfg.rewards.gait_threshold[1]) & (
                real_vel >= self.cfg.rewards.gait_threshold[0])
        self.mask_bound = (real_vel > self.cfg.rewards.gait_threshold[1])

        super()._post_physics_step_callback()

    def _init_buffers(self):
        super()._init_buffers()

    def _reset_root_states(self, env_ids):
        if self.cfg.terrain.mesh_type == 'gap_parkour':
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]

            self.env_list = self.env_class[env_ids]
            self.env_ids = env_ids[self.env_list != 17]
            self.root_states[self.env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(self.env_ids), 2),
                                                                   device=self.device)

        elif self.cfg.terrain.mesh_type == 'mix':
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

            # print('env_ids',env_ids)
            self.env_list = self.env_class[env_ids]
            self.env_ids_step = env_ids[self.env_list == 5]
            self.env_ids_gap = env_ids[self.env_list == 6]
            self.env_ids_other = env_ids[(self.env_list != 5) & (self.env_list != 6)]

            if len(self.env_ids_step) > 0:
                self.root_states[self.env_ids_step, 0:1] -= torch_rand_float(2., 3., (len(self.env_ids_step), 1),
                                                                             device=self.device)
            if len(self.env_ids_gap) > 0:
                self.root_states[self.env_ids_gap, 0:1] -= torch_rand_float(0, 0.4, (len(self.env_ids_gap), 1),
                                                                            device=self.device)
            if len(self.env_ids_other) > 0:
                self.root_states[self.env_ids_other, :2] += torch_rand_float(-0.5, 0.5, (len(self.env_ids_other), 2),
                                                                             device=self.device)

        else:
            # base position
            if self.custom_origins:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
                # xy position within 1m of the center
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:8] = torch_rand_float(0, 0.5, (len(env_ids), 1),
        #                                                   device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def _reward_motion_trot(self):
        # cosmetic penalty for motion
        rew1 = torch.sum(torch.abs(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [9, 10, 11]]), dim=1)
        rew2 = torch.sum(torch.abs(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
        rew = rew1 + rew2
        # rew[~self.mask_trot] *= 0

        rew[self.terrain_levels > self.cfg.env.env_gait] *= 0
        if self.terrain_adaptive_reward:
            if self.cfg.terrain.mesh_type in ["mix"]:
                rew[self.env_class == 5] *= 0.0
                rew[self.env_class == 6] *= 0
            elif self.cfg.terrain.mesh_type in ["gap_parkour"]:
                mask_step = (self.env_class == 1) | (self.env_class == 2)
                rew[mask_step] *= 1
                rew[self.env_class == 3] *= 0.1
                rew[self.env_class == 13] *= 0
                rew[self.env_class == 14] *= 0
                rew[self.env_class == 15] *= 0
                rew[self.env_class == 16] *= 1
                rew[self.env_class == 17] *= 1
        else:
            pass
        mask = self.common_step_counter > 2e4
        rew[mask] *= 0
        return rew

    def _reward_motion_bound(self):
        # cosmetic penalty for motion
        rew1 = torch.sum(torch.abs(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [3, 4, 5]]), dim=1)
        rew2 = torch.sum(torch.abs(self.dof_pos[:, [9, 10, 11]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
        rew = rew1 + rew2
        rew[~self.mask_bound] *= 0
        return rew

    def _reward_motion_pace(self):
        # cosmetic penalty for motion
        rew1 = torch.sum(torch.abs(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
        rew2 = torch.sum(torch.abs(self.dof_pos[:, [9, 10, 11]] - self.dof_pos[:, [3, 4, 5]]), dim=1)
        rew = rew1 + rew2
        rew[~self.mask_pace] *= 0
        return rew

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        rew = torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                        dim=1)

        if self.terrain_adaptive_reward:
            if self.cfg.terrain.mesh_type in ["mix"]:
                rew[self.env_class == 5] *= 0.0
                rew[self.env_class == 6] *= 0
            elif self.cfg.terrain.mesh_type in ["gap_parkour"]:
                mask_step = (self.env_class == 1) | (self.env_class == 2)
                rew[mask_step] *= 0.2
                rew[self.env_class == 3] *= 0.1
                rew[self.env_class == 13] *= 0.2
                rew[self.env_class == 14] *= 10
                rew[self.env_class == 15] *= 10
                rew[self.env_class == 16] *= 0.2
                rew[self.env_class == 17] *= 0

        return rew

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

        return rew

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel = self.base_lin_vel[:, :2].clone()

        lin_vel_clip = torch.tensor(self.cfg.rewards.lin_vel_clip,
                                    dtype=self.commands.dtype,
                                    device=self.commands.device)

        large_value = torch.tensor(1e5,
                                   dtype=self.commands.dtype,
                                   device=self.commands.device)

        lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0,
                                          large_value,
                                          self.commands[:, :2] + lin_vel_clip)

        lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0,
                                          -large_value,
                                          self.commands[:, :2] - lin_vel_clip)

        clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - clip_lin_vel), dim=1)
        tracking_sigma = torch.tensor(self.cfg.rewards.tracking_sigma,
                                      dtype=self.commands.dtype,
                                      device=self.commands.device)

        rew = torch.exp(-lin_vel_error / tracking_sigma)

        return rew


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        rew = torch.square(self.base_lin_vel[:, 2])
        # print(self.env_class)
        if self.terrain_adaptive_reward:
            if self.cfg.terrain.mesh_type in ["mix"]:
                rew[self.env_class == 5] *= 0.25
                rew[self.env_class == 6] *= 0.25
            elif self.cfg.terrain.mesh_type in ["gap_parkour"]:
                mask_step = (self.env_class == 1) | (self.env_class == 2)
                rew[mask_step] *= 0.25
                rew[self.env_class == 3] *= 0.05
                rew[self.env_class == 4] *= 0.05
                rew[self.env_class == 7] *= 0.05

                rew[self.env_class == 13] *= 0.1
                rew[self.env_class == 14] *= 1.0
                rew[self.env_class == 15] *= 0.01
                rew[self.env_class == 16] *= 1.0
                rew[self.env_class == 17] *= 1.0
        else:
            pass

        return rew

    def _reward_orientation(self):
        # Penalize non flat base orientation
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        if self.terrain_adaptive_reward:
            if self.cfg.terrain.mesh_type in ["mix"]:
                rew[self.env_class == 5] *= 0.25
                rew[self.env_class == 1] *= 1
            elif self.cfg.terrain.mesh_type in ["gap_parkour"]:
                mask_step = (self.env_class == 1) | (self.env_class == 2)
                rew[mask_step] *= 0.25
                rew[self.env_class == 3] *= 1.0
                rew[self.env_class == 4] *= 1.0
                rew[self.env_class == 13] *= 0.01
                rew[self.env_class == 14] *= 1.0
                rew[self.env_class == 15] *= 0.01
                rew[self.env_class == 16] *= 1.0
                rew[self.env_class == 17] *= 0.0
        else:
            pass

        return rew
    def _reward_nav_point(self):
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        if not self.terrain_adaptive_reward:
            return rew
        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            mask = self.env_class == 17
            if not torch.any(mask):
                return rew

            current_levels = self.terrain_levels[mask].float()
            current_levels = torch.clamp(current_levels, min=1.0)

            robot_pos = self.root_states[mask]

            terrain_level = torch.from_numpy(self.terrain.goals_narrow.astype(np.float32)).to(self.device)
            target_pos = terrain_level[self.terrain_levels[mask].long()]

            distance = torch.norm(robot_pos[:, :1] - target_pos[:, :1], dim=1)
            base_rew = torch.exp(-distance / 10)

            total_rew = ((current_levels - 1) + base_rew) * 1
            rew[mask] = total_rew

        return rew

    def _reward_nav_in_command_direction(self):
        """Penalize standing still with move command; reward moving in command direction."""
        command = self.commands[:, 0]
        command_magnitude = torch.abs(command)

        velocity = self.base_lin_vel[:, 0]
        velocity_magnitude = torch.abs(velocity)

        stop_penalty = (velocity_magnitude < 0.2) * (command_magnitude > 0.2) * 1.0

        direction_reward = velocity / velocity_magnitude * 1

        rew = -stop_penalty + direction_reward

        if self.terrain_adaptive_reward:
            rew[self.env_class != 17] *= 0

        return rew

    def _reward_nav_collision(self):
        rew = torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_narrow_indices, :], dim=-1) > 0.1), dim=1)
        if self.terrain_adaptive_reward and self.cfg.terrain.mesh_type in ["gap_parkour"]:
            rew[self.env_class != 17] *= 0
        return rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
                torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt

        # if self.cfg.terrain.mesh_type in ["gap_parkour"]:
        #     mask_step = (self.env_class == 1) | (self.env_class == 2)
        #     rew_airTime[mask_step] *= 0.5
        #     rew_airTime[self.env_class == 3] *= 0.5
        #     rew_airTime[self.env_class == 13] *= 0.25
        #     rew_airTime[self.env_class == 14] *= 2
        #     rew_airTime[self.env_class == 15] *= 0.25
        #     rew_airTime[self.env_class == 16] *= 2
        #     rew_airTime[self.env_class == 17] *= 2
        return rew_airTime



    def _reward_feet_edge(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)

        feet_pos_xy = ((self.rigid_body_state[:, self.feet_indices,
        :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        # print('x_edge_mask', feet_at_edge)
        self.feet_at_edge = contact_filt & feet_at_edge

        rew = (self.terrain_levels > 0) * torch.sum(self.feet_at_edge, dim=-1)
        rew = rew.to(torch.float32)
        if self.terrain_adaptive_reward and self.cfg.terrain.mesh_type in ["gap_parkour"]:
            mask_step = ((self.env_class == 0) | (self.env_class == 1) | (self.env_class == 2) |
                         (self.env_class == 17) | (self.env_class == 15) | (self.env_class == 16))
            rew[mask_step] *= 0

        return rew


    def _resample_easy_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _resample_hard_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["new_lin_vel_x"][0],
                                                     self.command_ranges["new_lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["new_lin_vel_y"][0],
                                                     self.command_ranges["new_lin_vel_y"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["new_heading"][0],
                                                         self.command_ranges["new_heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["new_ang_vel_yaw"][0],
                                                         self.command_ranges["new_ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return

        env_list = self.env_class[env_ids]

        if self.cfg.terrain.mesh_type in ["mix"]:
            easy_mask = (env_list == 0) | (env_list == 1) | (env_list == 2) | (env_list == 3) | (env_list == 4)
        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            easy_mask = (env_list == 0) | (env_list == 1) | (env_list == 2)

        easy_ids = env_ids[easy_mask]
        hard_ids = env_ids[~easy_mask]

        if easy_ids.numel() > 0 or hard_ids.numel() > 0:
            if self.common_step_counter < 2e6:
                self._resample_easy_commands(easy_ids)
                self._resample_hard_commands(hard_ids)
            else:
                self._resample_easy_commands(easy_ids)
                self._resample_hard_commands(hard_ids)
        # set small commands to zero
        if self.cfg.commands.zero_command and env_ids.numel() > 0:
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            adjust_dis = to_torch((self.cfg.init_state.pos[0:2]), device=self.device)
            real_dis = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
            distance = torch.norm(real_dis - adjust_dis, dim=1)
        else:
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        if self.cfg.terrain.mesh_type in ["mix"]:
            self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            self.joint_pos_target = actions_scaled + self.default_dof_pos

            if self.cfg.domain_rand.randomize_action_latency:
                self.joint_pos_target = self.joint_pos_target + self.motor_offsets

                # self.joint_pos_target  = torch.clip(self.joint_pos_target , self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

                torques = (self.p_gains * (self.joint_pos_target - self.dof_pos)
                           - self.d_gains * self.dof_vel)
            else:
                torques = (self.p_gains * (self.joint_pos_target - self.dof_pos)
                           - self.d_gains * self.dof_vel)

            # scale the output
            torques = torques * self.motor_strengths
        elif control_type == "P_factors":
            if self.cfg.domain_rand.randomize_lag_timesteps:
                prev_action = self.lag_buffer[:, 1:, :].clone()
                self.lag_buffer[
                    :, :-1, :] = prev_action  # To copy the historical images (except the last one) back to self.image_buf
                self.lag_buffer[
                    :, -1, :] = actions_scaled  # To replace the current image with the last image from self.image_buf
                index = random.randint(0, self.cfg.domain_rand.added_lag_timesteps)
                self.joint_pos_target = self.lag_buffer[:, index, :] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos
            self.joint_pos_target = torch.clip(self.joint_pos_target, self.dof_pos_limits[:, 0],
                                               self.dof_pos_limits[:, 1])

            torques = (self.p_gains * (self.joint_pos_target - self.dof_pos + self.motor_offsets)
                       - self.d_gains * self.dof_vel)

            # scale the output
            torques = torques * self.motor_strengths


        elif control_type == "V":
            torques = (
                    self.p_gains * (actions_scaled - self.dof_vel)
                    - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        # return torch.clip(torques, -0.9 * self.torque_limits, 0.9 * self.torque_limits)