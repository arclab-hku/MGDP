from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.envs import LeggedRobot
from .legged_robot_config_baseline import LeggedRobotBaseCfg
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
import cv2, os

class Legged_terrains(LeggedRobot):
    cfg: LeggedRobotBaseCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.morphology_const_info_dict = {
            'payload': (0, 1),  # base
            'friction': (1, 2),
            'stiffness': (2, 3),  # p_gain
            'damping': (3, 4),  # d_gain
            'motor_strengths': (4, 16),  # motor strengths
            'motor_offsets': (16, 28),  # motor offsets
            'com': (28, 30),
            'limb_mass': (30, 33),
            'restitution': (33, 34),
        }

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.height_clip = -0.2

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.cfg.control.control_type == "P_factors" :
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_buffer[env_ids, :, :] = 0

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          self.cfg.commands.min_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)


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

    def _reset_root_states1(self, env_ids):
        if self.cfg.terrain.mesh_type =='gap_parkour' :
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]

            self.root_states[env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device)

        elif self.cfg.terrain.mesh_type == 'mix':
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

            # print('env_ids',env_ids)
            self.env_list = self.env_class[env_ids]
            self.env_ids_step = env_ids[self.env_list == 5]
            self.env_ids_gap = env_ids[self.env_list == 6]
            self.env_ids_other = env_ids[(self.env_list != 5) & (self.env_list != 6)]

            if len(self.env_ids_step)>0:
                self.root_states[self.env_ids_step, 0:1] -= torch_rand_float(2., 3., (len(self.env_ids_step), 1), device=self.device)
            if len(self.env_ids_gap) > 0:
                self.root_states[self.env_ids_gap, 0:1] -= torch_rand_float(0, 0.4, (len(self.env_ids_gap), 1), device=self.device)
            if len(self.env_ids_other) > 0:
                self.root_states[self.env_ids_other, :2] += torch_rand_float(-0.5, 0.5, (len(self.env_ids_other), 2), device=self.device)

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
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def step(self, env_ids):
        super().step(env_ids)
        self.obs_dict['proprio_hist'] = self.obs_history_buf

    def _create_trimesh(self):
        super()._create_trimesh()
        if self.cfg.terrain.mesh_type in ["mix"]:
            self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows,
                                                                           self.terrain.tot_cols).to(self.device)

    def _create_gap_parkour_trimesh(self):
        """ Adds a triangle meshes terrains to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        # tm_params.nb_vertices = self.terrain.vertices.shape[0]
        # tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        if self.cfg.terrain.add_air_beam:
            # add air beams
            for i in range(len(self.terrain.beam_vertices)):
                # add increment of the triangles
                self.terrain.beam_triangles[i] = self.terrain.beam_triangles[i] + self.terrain.vertices.shape[0]
                self.terrain.vertices = np.concatenate((self.terrain.vertices, self.terrain.beam_vertices[i]), axis=0)
                self.terrain.triangles = np.concatenate((self.terrain.triangles, self.terrain.beam_triangles[i]), axis=0)

        if self.cfg.terrain.add_air_stone:
            # add air stones
            for i in range(len(self.terrain.stone_triangles)):
                # add increment of the triangles
                self.terrain.stone_triangles[i] = self.terrain.stone_triangles[i] + self.terrain.vertices.shape[0]
                self.terrain.vertices = np.concatenate((self.terrain.vertices, self.terrain.stone_vertices[i]), axis=0)
                self.terrain.triangles = np.concatenate((self.terrain.triangles, self.terrain.stone_triangles[i]), axis=0)

        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        # print('tm_params', self.terrain.vertices.shape, self.terrain.triangles.shape)
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)
        if self.cfg.terrain.mesh_type == "gap_parkour":
            self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows,
                                                                           self.terrain.tot_cols).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises:
            NameError: [description]
        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain meshes type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        real_height = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return real_height

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.num_histroy_obs*self.cfg.env.num_observations, device=self.device, dtype=torch.float)
        self.obs_dict['proprio_hist'] = self.obs_history_buf
        # if self.cfg.privInfo.enableForce:
        #     self.disturbance_force = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel

    def compute_observations(self):
        super().compute_observations()

        if self.cfg.privInfo.enablePayload:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 0:1],
                                                 ), dim=-1)

        if self.cfg.privInfo.enableFriction:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 1:2],
                                                 ), dim=-1)

        if self.cfg.privInfo.enableStiffnessDamping:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 2:4],
                                                 ), dim=-1)
        if self.cfg.privInfo.enableMotorStrength:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 4:16],
                                                 ), dim=-1)
        if self.cfg.privInfo.enablemMotorOffsets:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 16:28],
                                                 ), dim=-1)
        # print('3', self.privileged_obs_buf.shape )

        if self.cfg.privInfo.enableCom:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 28:30],
                                                 ), dim=-1)

        if self.cfg.privInfo.enableLimb_mass:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 30:33],
                                                 ), dim=-1)

        if self.cfg.privInfo.enableForce:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.disturbance_force,
                                                 ), dim=-1)

        if self.cfg.privInfo.enableFootContact:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
            contact_filt = torch.logical_or(contact, self.last_contacts)
            self.last_contacts = contact
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 contact_filt,
                                                 ), dim=-1)
        if self.cfg.privInfo.enableFootHeight:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.foot_height,
                                                 ), dim=-1)

        if self.cfg.privInfo.enableMaxFootHeight:
            max_foot = torch.max(self.foot_height, axis=1)[0]
            max_foot = max_foot.reshape(self.num_envs, 1)

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, max_foot), dim=-1)

        if self.cfg.privInfo.enableMeasuredHeight:
            if self.cfg.terrain.mesh_type == 'gap_parkour':
                self.heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, -1,
                                     1.) * self.obs_scales.height_measurements
            else:
                # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, -1.,
                #                  1.) * self.obs_scales.height_measurements
                self.heights = torch.clip(self.root_states[:, 2].unsqueeze(1)  - self.measured_heights, -1,
                                     1.) * self.obs_scales.height_measurements
            if self.add_height_noise:
                heights_noise = (2 * torch.rand_like(self.heights) - 1) * self.noise_scale_vec_height
                self.heights = self.heights + heights_noise

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.heights), dim=-1)

            if self.cfg.env.measure_obs_heights:
                self.obs_buf = torch.cat((self.obs_buf, self.heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.env.use_history:
            # deal with normal observation, do sliding window
            prev_obs_buf =  self.obs_history_buf[:, 45:]
            # concatenate to get full history
            self.obs_history_buf = torch.cat([prev_obs_buf, self.obs_buf], dim=1)
            # print('obs_buf', self.obs_buf[1])  # 应该是 [10, 45]
            # print('his', self.obs_history_buf[1, :])  # 应该是 [10, 180]
            # print()
            at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(at_reset_env_ids) != 0:
                self.obs_history_buf[at_reset_env_ids, :] = self.obs_buf[at_reset_env_ids].unsqueeze(1).repeat(1, self.cfg.env.num_histroy_obs, 1) .view(len(at_reset_env_ids),-1)  # 形状变为 [N, 45 * num_repeats]

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
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
        # set small commands to zero
        if self.cfg.commands.zero_command:
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrains heights and randomly push urdf
        """
        self.env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(self.env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            if self.cfg.terrain.mesh_type in ["mix"]:
                self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            else:
                self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval_s == 1):
            self.disturbance_force = self._push_robots()

        if self.cfg.terrain.measure_feet_heights:
            # env_ids = None # None
            self.measured_FL_foot_heights = self._get_foot_heights(t=0,  num_foot_height_points=self.num_foot_height_points_FL_foot,
                                                                height_foot_points=self.height_FL_foot_points, env_ids=self.env_ids)
            self.measured_FR_foot_heights = self._get_foot_heights(t=1, num_foot_height_points=self.num_foot_height_points_FR_foot,
                                                                 height_foot_points=self.height_FR_foot_points, env_ids=self.env_ids)
            self.measured_RL_foot_heights = self._get_foot_heights(t=2, num_foot_height_points=self.num_foot_height_points_RL_foot,
                                                                 height_foot_points=self.height_RL_foot_points, env_ids=self.env_ids)
            self.measured_RR_foot_heights = self._get_foot_heights(t = 3, num_foot_height_points=self.num_foot_height_points_RR_foot,
                                                                 height_foot_points=self.height_RR_foot_points, env_ids=self.env_ids)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def foot_point(self, point_x, point_y):
        y = torch.tensor(point_y, device=self.device, requires_grad=False)
        x = torch.tensor(point_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        return grid_x, grid_y

    def draw_sparse_heatmap(self, pred_height, true_height, cell_size=50, gap=20):
        """
        离散显示每个高程点（放大球体尺寸，优化显示效果）
        """
        # 获取全局最大最小值，统一颜色映射范围
        all_values = np.concatenate([pred_height.flatten(), true_height.flatten()])
        vmin, vmax = -1, 1
        if vmax - vmin < 1e-8:
            vmin, vmax = 0, 1

        # 画布尺寸计算
        canvas_height = 17 * cell_size
        canvas_width = 2 * 11 * cell_size + gap
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # 白色背景

        # 绘制网格线（不变）
        for i in range(17 + 1):
            y = i * cell_size
            cv2.line(canvas, (0, y), (canvas_width, y), (200, 200, 200), 1)
        for j in range(11 + 1):
            x = j * cell_size
            cv2.line(canvas, (x, 0), (x, canvas_height), (200, 200, 200), 1)
        for j in range(11 + 1):
            x = 11 * cell_size + gap + j * cell_size
            cv2.line(canvas, (x, 0), (x, canvas_height), (200, 200, 200), 1)
        cv2.line(canvas, (11 * cell_size, 0), (11 * cell_size, canvas_height), (150, 150, 150), 2)
        cv2.line(canvas, (11 * cell_size + gap, 0), (11 * cell_size + gap, canvas_height), (150, 150, 150), 2)

        # 绘制每个点（重点修改球体大小）
        for i in range(17):
            for j in range(11):
                # 预测点（左列）
                pred_val = pred_height[i, j]
                pred_norm = (pred_val - vmin) / (vmax - vmin)
                pred_color = (
                    int(255 * (1 - pred_norm)),
                    int(255 * min(2 * pred_norm, 2 - 2 * pred_norm)),
                    int(255 * pred_norm)
                )
                # 预测点位置
                pred_x = j * cell_size + cell_size // 2
                pred_y = i * cell_size + cell_size // 2
                # 增大球体半径（从cell_size//4 改为 cell_size//3，约为网格的1/3）
                pred_radius = cell_size // 3
                cv2.circle(canvas, (pred_x, pred_y), pred_radius, pred_color, -1)  # 填充圆
                # 调整文字位置（避免被球体遮挡）
                # cv2.putText(canvas, f"{pred_val:.2f}",
                #             (pred_x - cell_size // 3, pred_y + cell_size // 2 - 5),  # 下移文字
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 稍大字体

                # 真实点（右列）
                true_val = true_height[i, j]
                true_norm = (true_val - vmin) / (vmax - vmin)
                true_color = (
                    int(255 * (1 - true_norm)),
                    int(255 * min(2 * true_norm, 2 - 2 * true_norm)),
                    int(255 * true_norm)
                )
                # 真实点位置
                true_x = 11 * cell_size + gap + j * cell_size + cell_size // 2
                true_y = i * cell_size + cell_size // 2
                # 同样增大半径
                true_radius = cell_size // 3
                cv2.circle(canvas, (true_x, true_y), true_radius, true_color, -1)
                # # 调整文字位置
                # cv2.putText(canvas, f"{true_val:.2f}",
                #             (true_x - cell_size // 3, true_y + cell_size // 2 - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # # 添加标题（不变）
        # cv2.putText(canvas, "Predicted Heights",
        #             (11 * cell_size // 2 - 80, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # cv2.putText(canvas, "True Heights",
        #             (11 * cell_size + gap + 11 * cell_size // 2 - 60, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return canvas

    def draw_sparse_heatmap1(self, pred_height, true_height, cell_size=60, gap=20):
        """
        颜色条放在主图左侧，高度缩短且文字清晰
        """
        # 1. 计算颜色映射范围
        all_values = np.concatenate([pred_height.flatten(), true_height.flatten()])
        vmin, vmax = -1, 1
        if vmax - vmin < 1e-8:
            vmin, vmax = 0, 1

        # 2. 生成主图（高程点对比）
        canvas_height = 17 * cell_size  # 主图高度
        canvas_width = 2 * 11 * cell_size + gap
        main_img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # 绘制主图网格线
        for i in range(17 + 1):
            y = i * cell_size
            cv2.line(main_img, (0, y), (canvas_width, y), (200, 200, 200), 1)
        for j in range(11 + 1):
            x = j * cell_size
            cv2.line(main_img, (x, 0), (x, canvas_height), (200, 200, 200), 1)
        for j in range(11 + 1):
            x = 11 * cell_size + gap + j * cell_size
            cv2.line(main_img, (x, 0), (x, canvas_height), (200, 200, 200), 1)
        cv2.line(main_img, (11 * cell_size, 0), (11 * cell_size, canvas_height), (150, 150, 150), 2)
        cv2.line(main_img, (11 * cell_size + gap, 0), (11 * cell_size + gap, canvas_height), (150, 150, 150), 2)

        # 绘制高程点
        for i in range(17):
            for j in range(11):
                # 预测点
                pred_val = pred_height[i, j]
                pred_norm = (pred_val - vmin) / (vmax - vmin)
                pred_color = (int(255 * (1 - pred_norm)), int(255 * min(2 * pred_norm, 2 - 2 * pred_norm)),
                              int(255 * pred_norm))
                pred_x = j * cell_size + cell_size // 2
                pred_y = i * cell_size + cell_size // 2
                cv2.circle(main_img, (pred_x, pred_y), cell_size // 3, pred_color, -1)

                # 真实点
                true_val = true_height[i, j]
                true_norm = (true_val - vmin) / (vmax - vmin)
                true_color = (int(255 * (1 - true_norm)), int(255 * min(2 * true_norm, 2 - 2 * true_norm)),
                              int(255 * true_norm))
                true_x = 11 * cell_size + gap + j * cell_size + cell_size // 2
                true_y = i * cell_size + cell_size // 2
                cv2.circle(main_img, (true_x, true_y), cell_size // 3, true_color, -1)

        # 3. 生成颜色条（高度缩短，放在左侧）
        bar_height = canvas_height -2  # 颜色条高度 = 主图高度的一半
        bar_width = 90
        colorbar = np.ones((bar_height, bar_width, 3), dtype=np.uint8) * 255

        # 绘制颜色条渐变
        for i in range(bar_height):
            norm = i / (bar_height - 1)
            val = vmax - norm * (vmax - vmin)
            val_norm = (val - vmin) / (vmax - vmin)
            r = int(255 * (1 - val_norm))
            g = int(255 * min(2 * val_norm, 2 - 2 * val_norm))
            b = int(255 * val_norm)
            cv2.line(colorbar, (0, i), (bar_width, i), (b, g, r), 1)

        # 绘制刻度和文字（左侧布局，调整位置）
        tick_positions = [0, bar_height // 4, bar_height // 2, 3 * bar_height // 4, bar_height - 1]
        tick_values = [vmax, 0.75, 0.45, 0.15, vmin]
        for y, val in zip(tick_positions, tick_values):
            cv2.line(colorbar, (0, y), (8, y), (0, 0, 0), 3)  # 刻度线左移

            text = f"{val:.2f}"
            if y == 0:  # 顶部(1.00)
                cv2.putText(
                    colorbar,
                    text,
                    (20, y + 20),  # 右移文字，避免被刻度线遮挡
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )
            elif y == bar_height - 1:  # 底部(-1.00)
                cv2.putText(
                    colorbar,
                    text,
                    (20, y - 5),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )
            else:  # 中间刻度
                cv2.putText(
                    colorbar,
                    text,
                    (20, y + 15),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        # 4. 拼接（颜色条在左，主图在右，垂直居中）
        colorbar_container = np.ones((canvas_height, bar_width, 3), dtype=np.uint8) * 255
        offset_y = (canvas_height - bar_height) // 2
        colorbar_container[offset_y:offset_y + bar_height, :, :] = colorbar

        combined_img = np.hstack((colorbar_container, main_img))  # 左-右拼接
        return combined_img

    def _init_FL_foot_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_foot_height_points, 3)
        """
        grid_x_FL_foot, grid_y_FL_foot = self.foot_point(self.cfg.terrain.FL_foot_measured_points_x,
                                                         self.cfg.terrain.FL_foot_measured_points_y)

        self.num_foot_height_points_FL_foot = grid_x_FL_foot.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points_FL_foot, 3, device=self.device,
                             requires_grad=False)

        points[:, :, 0] = grid_x_FL_foot.flatten()
        points[:, :, 1] = grid_y_FL_foot.flatten()

        return points

    def _init_FR_foot_height_points(self):
        grid_x_FR_foot, grid_y_FR_foot = self.foot_point(self.cfg.terrain.FR_foot_measured_points_x,
                                                         self.cfg.terrain.FR_foot_measured_points_y)

        self.num_foot_height_points_FR_foot = grid_x_FR_foot.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points_FR_foot, 3, device=self.device,
                             requires_grad=False)

        points[:, :, 0] = grid_x_FR_foot.flatten()
        points[:, :, 1] = grid_y_FR_foot.flatten()
        return points

    def _init_RL_foot_height_points(self):
        grid_x_RL_foot, grid_y_RL_foot = self.foot_point(self.cfg.terrain.RL_foot_measured_points_x,
                                                         self.cfg.terrain.RL_foot_measured_points_y)

        self.num_foot_height_points_RL_foot = grid_x_RL_foot.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points_RL_foot, 3, device=self.device,
                             requires_grad=False)

        points[:, :, 0] = grid_x_RL_foot.flatten()
        points[:, :, 1] = grid_y_RL_foot.flatten()
        return points

    def _init_RR_foot_height_points(self):
        grid_x_RR_foot, grid_y_RR_foot = self.foot_point(self.cfg.terrain.RR_foot_measured_points_x,
                                                         self.cfg.terrain.RR_foot_measured_points_y)

        self.num_foot_height_points_RR_foot = grid_x_RR_foot.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points_RR_foot, 3, device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x_RR_foot.flatten()
        points[:, :, 1] = grid_y_RR_foot.flatten()
        return points

    def _get_foot_heights(self, t=None, num_foot_height_points=None, height_foot_points=None, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, num_foot_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrains meshes type 'none'")

        foot_pos = self.foot_pos[:, t, :]

        points = quat_apply_yaw(self.base_quat.repeat(1, num_foot_height_points), height_foot_points) + (
            foot_pos).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _draw_camera_pos_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # draw height lines
        if self.cfg.terrain.measure_heights:
            sphere_geom = gymutil.WireframeSphereGeometry(0.04, 4, 4, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()

                x = base_pos[0] + 0.35
                y = base_pos[1]
                z = base_pos[2] + 0.05

                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrains the origins are defined by the terrains platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["trimesh"]:
            self.custom_origins = self.cfg.terrain.custom_origins
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put urdf at the origins defined by the terrains
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)

            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

        if self.cfg.terrain.mesh_type in [ "gap_parkour"]:
            self.custom_origins = self.cfg.terrain.custom_origins
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

            # put urdf at the origins defined by the terrains
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)

            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            # self.step_gap_flags = (self.env_class == 1) | (self.env_class == 13)
            # self.step_gap_ids = self.step_gap_flags.nonzero(as_tuple=False).flatten()

        elif self.cfg.terrain.mesh_type in ["mix"]:
            self.custom_origins = self.cfg.terrain.custom_origins
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

            # put urdf at the origins defined by the terrains
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)

            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
            # self.step_gap_flags = (self.env_class == 5) | (self.env_class == 6)
            # self.step_gap_ids = self.step_gap_flags.nonzero(as_tuple=False).flatten()

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of urdf
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def post_physics_step(self):
        super().post_physics_step()
        if self.viewer and self.enable_viewer_sync:
            if self.debug_viz == 'base':
                self._draw_base_vis()
            elif self.debug_viz == 'feet':
                self._draw_foot_vis()
            elif self.debug_viz == 'xyz':
                self._draw_init_vis()
            elif self.debug_viz == 'camera':
                self._draw_camera_pos_vis()
            elif self.debug_viz == 'both':
                self._draw_base_vis()
                self._draw_init_vis()
            else:
                pass

    def _draw_foot_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        np.set_printoptions(precision=4)
        # draw height lines
        if self.cfg.terrain.measure_feet_heights:
            FL_foot_sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 0, 0))
            FR_foot_sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 1, 0))
            RL_foot_sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 0))
            RR_foot_sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 1))

            for i in range(self.num_envs):
                for t in range(4):
                    if t == 0:
                        heights = self.measured_FL_foot_heights[i].cpu().numpy()
                        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                                       self.height_FL_foot_points[i]).cpu().numpy()
                    elif t == 1:
                        heights = self.measured_FR_foot_heights[i].cpu().numpy()
                        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                                       self.height_FR_foot_points[i]).cpu().numpy()
                    elif t == 2:
                        heights = self.measured_RL_foot_heights[i].cpu().numpy()
                        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                                       self.height_RL_foot_points[i]).cpu().numpy()
                    elif t == 3:
                        heights = self.measured_RR_foot_heights[i].cpu().numpy()
                        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                                       self.height_RR_foot_points[i]).cpu().numpy()

                    if self.foot_pos.shape == torch.Size([4, 3]):
                        foot_pos = self.foot_pos[t, :].cpu().numpy()
                    else:
                        foot_pos = self.foot_pos[i, t, :].cpu().numpy()
                    for j in range(heights.shape[0]):
                        x = height_points[j, 0] + foot_pos[0]
                        y = height_points[j, 1] + foot_pos[1]
                        z = heights[j]

                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        if t == 0:
                            gymutil.draw_lines(FL_foot_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                        elif t == 1:
                            gymutil.draw_lines(FR_foot_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                        elif t == 2:
                            gymutil.draw_lines(RL_foot_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                        elif t == 3:
                            gymutil.draw_lines(RR_foot_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def create_sim(self):
        """Creates simulation, terrains and evironments"""
        super().create_sim()
        mesh_type = self.cfg.terrain.mesh_type
        if self.cfg.terrain.mesh_type == "new_parkour":
            self._create_terrain()
        else:
            if mesh_type in [ "trimesh", "gap_parkour", "mix"]:
                self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            if mesh_type == "plane":
                self._create_ground_plane()
            elif mesh_type == "trimesh":
                self._create_trimesh()
            elif mesh_type == "mix":
                self._create_trimesh()
            elif mesh_type == "gap_parkour":
                self._create_gap_parkour_trimesh()
            elif mesh_type is not None:
                raise ValueError(
                    "Terrain meshes type not recognised. Allowed types are [None, plane, trimesh, gap_parkour, mix]"
                )

        self._create_envs()


    def _update_morph_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.morphology_const_info_dict[name]
        if type(value) is list:
            value = to_torch(value, dtype=torch.float, device=self.device)
        if type(lower) is list or type(upper) is list:
            lower = to_torch(lower, dtype=torch.float, device=self.device)
            upper = to_torch(upper, dtype=torch.float, device=self.device)
        if lower is not None and upper is not None:
            value = (value - lower) / (upper - lower)
        self.morph_priv_info_buf[env_id, s:e] = value


    def _process_rigid_body_props(self, props, env_id, robot_type_id):
        self.default_body_mass = props[0].mass
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass = self.default_body_mass + np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, 4):
                props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        if self.cfg.privInfo.enablePayload:
            link_mass = 4 * (props[1].mass + props[2].mass + props[3].mass)
            real_mass = link_mass + props[0].mass
            self._update_morph_priv_buf(env_id=env_id, name='payload', value=real_mass, lower=0, upper=20)

            # self._update_morph_priv_buf(env_id=env_id, name='payload', value=props[0].mass, lower=0, upper=20)
        if self.cfg.privInfo.enableLimb_mass:
            limb_mass = [props[1].mass, props[2].mass, props[3].mass]  # take FL's hip, thigh, calf for example

            self._update_morph_priv_buf(env_id=env_id, name='limb_mass', value=limb_mass, lower=0, upper=20)
        if self.cfg.domain_rand.randomize_com:
            center = self.cfg.domain_rand.added_com_range
            com = [np.random.uniform(center[0], center[1]), np.random.uniform(center[0], center[1])]
            props[0].com.x, props[0].com.y = com
        return props

    def _process_rigid_shape_props(self, props, env_id):
        friction_range = self.cfg.domain_rand.added_friction_range
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            if self.cfg.privInfo.enableFriction:
                self._update_morph_priv_buf(env_id=env_id, name='friction', value=props[0].friction,
                                        lower=friction_range[0], upper=friction_range[1])

        return props


    def _process_dof_props(self, props, env_id, robot_type_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_actions, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]

                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            for i in range(self.num_dofs):
                name = self.dof_names[i]
                angle = self.cfg.init_state.default_joint_angles[name]
                self.default_dof_pos[i] = angle
                found = False
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # self.Kp
                        self.d_gains[i] = self.cfg.control.damping[dof_name]  # self.Kd
                        found = True
                if not found:
                    self.p_gains[i] = 0.0
                    self.d_gains[i] = 0.0
                    if self.cfg.control.control_type in ["P", "P_factors", "V"]:
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
            self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "P_factors" or self.cfg.domain_rand.randomize_action_latency:
            rand_motor_strength = np.random.uniform(self.cfg.domain_rand.added_motor_strength[0],
                                                    self.cfg.domain_rand.added_motor_strength[1])

            self.motor_strengths[env_id][:] = rand_motor_strength
            rand_motor_offset = np.random.uniform(self.cfg.domain_rand.added_motor_offset[0],
                                                  self.cfg.domain_rand.added_motor_offset[1])
            self.motor_offsets[env_id][:] = rand_motor_offset


        return props


    # ------------ _change_cmds functions----------------
    def _change_cmds(self, vx, vy, vang):
        # change command_ranges with the input
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = vang
        # self.commands[:, 3] = heading


