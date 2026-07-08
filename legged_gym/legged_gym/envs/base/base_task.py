import sys
from termcolor import cprint
from isaacgym import gymtorch, gymapi, gymutil

import numpy as np
import torch

import gym
from gym import spaces
class BaseTask():
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        cprint(f"sim_device: {sim_device}", 'red', attrs=['bold'])

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if self.sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device =  self.sim_device
        else:
            self.device = 'cpu'
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = cfg.env.graphics_device_num

        cprint(f"graphics_device: {self.graphics_device_id}", 'green', attrs=['bold'])
        cprint(f"device_count: {torch.cuda.device_count()}", 'yellow', attrs=['bold'])

        self.cfg = cfg
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        self._allocate_buffers()  # history specific buffers

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            cprint('Enable Visualization', 'green', attrs=['bold'])
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        else:
            self.viewer = None

        self.obs_dict = {}

    def get_observations(self):
        return self.obs_dict

    def reset_idx(self, env_ids):
        """Reset selected urdf"""
        raise NotImplementedError

    def reset(self):
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        env_ids = self.reset_buf.nonzero().squeeze(-1)
        self.reset_idx(env_ids)
        zero_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        # step the simulator
        self.obs_dict, _, _, _ = self.step(zero_actions)
        return self.obs_dict

    def step(self, actions):
        raise NotImplementedError

    def create_sim(self):
        """Creates simulation, terrains and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

    def _allocate_buffers(self):
        # additional buffer
        # self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.num_histroy_obs, self.num_obs), device=self.device, dtype=torch.float)
        self.morph_priv_info_buf = torch.zeros(self.num_envs, self.cfg.env.num_env_morph_priv_obs, device=self.device, dtype=torch.float)

    def render(self, sync_frame_time=True):
        if self.cfg.camera.camera_res is not None:
            self.gym.render_all_camera_sensors(self.sim)

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

        if self.cfg.camera.camera_res is not None or self.viewer:
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        if self.viewer:
            # step graphics
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)