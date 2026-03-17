from isaacgym.torch_utils import *
import torch
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs import LeggedRobot
from .legged_robot_config_baseline import LeggedRobotBaseCfg, LeggedRobotBaseCfgPPO

class Legged_rewards(LeggedRobot):
    cfg: LeggedRobotBaseCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.height_clip = -0.1
    def _reward_motion_trot(self):
        # cosmetic penalty for motion
        rew1 = torch.sum(torch.abs(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [9, 10, 11]]), dim=1)
        rew2 = torch.sum(torch.abs(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [6, 7, 8]]), dim=1)
        rew = rew1 + rew2


        rew[self.terrain_levels > 3] *= 0
        return rew

    def _reward_motion_base(self):
        # cosmetic penalty for motion
        rew = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        # print('dof_pos', self.dof_pos)
        # print('default_dof_pos', self.default_dof_pos)
        # rew = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

        rew[self.terrain_levels > 5] *= 0
        return rew

    def _reward_hip_motion(self):
        # cosmetic penalty for hip motion
        rew = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

        rew[self.terrain_levels > 5] *= 0
        return rew

    def _reward_thigh_motion(self):
        # cosmetic penalty for hip motion
        rew = torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - self.default_dof_pos[:, [1, 4, 7, 10]]), dim=1)
        rew[self.terrain_levels > 3] *= 0
        return rew

    def _reward_calf_motion(self):
        # cosmetic penalty for hip motion
        rew = torch.sum(torch.abs(self.dof_pos[:, [2, 5, 8, 11]] - self.default_dof_pos[:, [2, 5, 8, 11]]), dim=1)
        rew[self.terrain_levels > 3] *= 0
        return rew
        ############## Motion Functions ########

    def _reward_f_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3]] - self.default_dof_pos[:, [0, 3]]), dim=1)

    def _reward_r_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [6, 9]] - self.default_dof_pos[:, [6, 9]]), dim=1)

    def _reward_f_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4]] - self.default_dof_pos[:, [1, 4]]), dim=1)

    def _reward_r_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [7, 10]] - self.default_dof_pos[:, [7, 10]]), dim=1)

    def _reward_f_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5]] - self.default_dof_pos[:, [2, 5]]), dim=1)

    def _reward_r_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [8, 11]] - self.default_dof_pos[:, [8, 11]]), dim=1)

        ############## Dream ########

    def _reward_power_joint(self):
        r = torch.norm(self.torques[:, ], p=1, dim=1) * torch.norm(self.dof_vel[:, ], p=1, dim=1)
        return r

    def _reward_smoothness(self):
        return torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_actions_2), dim=1)

    def _reward_power_distribution(self):
        r = torch.mul(self.torques[:, ], self.dof_vel[:, ])
        d = torch.sum(torch.abs(r), dim=1)
        return d

    def _reward_feet_step(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_states_3d = rb_states.reshape(self.num_envs, -1, rb_states.shape[-1])
        feet_heights = rb_states_3d[
                       :, self.feet_indices, 2
                       ]  # proper way to get feet heights, don't use global feet ind
        feet_heights = feet_heights.view(-1)

        xy_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :2], dim=2
        ).view(-1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].view(-1)
        z_forces = torch.abs(z_forces)

        contact = torch.logical_or(
            self.contact_forces[:, self.feet_indices, 2] > 1.0,
            self.contact_forces[:, self.feet_indices, 1] > 1.0,
        )
        contact = torch.logical_or(
            contact, self.contact_forces[:, self.feet_indices, 0] > 1.0
        )

        last_contacts = contact
        xy_forces[feet_heights < 0.05] = 0
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, 4).sum(dim=1)
        z_ans[z_ans > 1] = 1

        return z_ans

    # ***************** energy disspation ***************
    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_motion(self):
        # cosmetic penalty for motion
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _draw_contact_polygon(self, contact_state, env_handle):
        contact_num = contact_state.shape[0]
        if contact_num >= 2:
            if contact_num == 4:  # switch the order of rectangle
                contact_state = contact_state[[0, 1, 3, 2], :]
            polygon_start = contact_state[0].cpu().numpy()

            width, n_lines = 0.01, 10  # make it thicker
            polygon_starts = []
            for i_line in range(n_lines):
                polygon_starts.append(polygon_start.copy())
                polygon_start += np.array([0, 0, width / n_lines])
            for i_feet in range(contact_num):
                polygon_end = contact_state[(i_feet + 1) % contact_num, :].cpu().numpy()

                polygon_ends = []
                polygon_vecs = []
                for i_line in range(n_lines):
                    polygon_ends.append(polygon_end.copy())
                    polygon_end += np.array([0, 0, width / n_lines])
                    polygon_vecs.append(
                        [polygon_starts[i_line][0], polygon_starts[i_line][1], polygon_starts[i_line][2],
                         polygon_ends[i_line][0], polygon_ends[i_line][1], polygon_ends[i_line][2]])
                self.gym.add_lines(self.viewer, env_handle, n_lines,
                                   polygon_vecs,
                                   n_lines * [0.85, 0.1, 0.1])

                polygon_starts = polygon_ends

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim=1)

    def _reward_alive(self):
        return 1.

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim=1)

    def _reward_feet_height(self):
        # self.re_feet = torch.zeros_like(self.foot_height)
        # for i in range(0, 4):
        #     real_foot_height = torch.mean(self.foot_height[:, i].unsqueeze(1) - self.measured_heights, dim=1)

        min_foot_height, _ = torch.min(self.foot_height, dim=1)
        foot_cutoff = (min_foot_height < -0.1)
        foot_reward = -1 * foot_cutoff
        #     self.re_feet[:, i] = torch.square(real_foot_height - self.cfg.rewards.foot_height_target)
        # b = torch.sum(torch.square(self.re_feet[:, ]), dim=1)
        # foot_reward = torch.exp(-10 * b)
        # print(foot_reward)
        return foot_reward

    def _reward_feet_clearance(self):
        # 从self.foot_vel中提取x和y方向的速度分量,计算水平速度
        vf_xy = torch.sqrt(self.foot_vel[:, :, 0] ** 2 + self.foot_vel[:, :, 1] ** 2)

        max_FL_foot_heights, max_FL_foot_index = torch.max(self.measured_FL_foot_heights, dim=1, keepdim=True)
        max_FR_foot_heights, max_FR_foot_index = torch.max(self.measured_FR_foot_heights, dim=-1, keepdim=True)
        max_RL_foot_heights, max_RL_foot_index = torch.max(self.measured_RL_foot_heights, dim=-1, keepdim=True)
        max_RR_foot_heights, max_RR_foot_index = torch.max(self.measured_RR_foot_heights, dim=-1, keepdim=True)

        max_foot_heights = torch.cat((max_FL_foot_heights, max_FR_foot_heights,
                                      max_RL_foot_heights, max_RR_foot_heights), dim=-1)
        print('max_foot_heights', max_foot_heights, max_foot_heights.shape)
        max_air_feet = torch.sum(
            torch.square(max_foot_heights + self.cfg.rewards.foot_height_target - self.foot_height), dim=1)
        print('max_air_feet', max_air_feet.shape, vf_xy.shape)
        rew_feet_clearance = torch.sum(torch.mul(max_air_feet, vf_xy), dim=1)
        print('rew_feet_clearance', rew_feet_clearance)
        return rew_feet_clearance

    def _reward_feet_first_center(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)

        F_index = [1, 3, 5, 7]
        R_index = [1, 3, 5, 7]

        center_count_FL_foot = torch.where(self.measured_FL_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_FR_foot = torch.where(self.measured_FR_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_RL_foot = torch.where(self.measured_RL_foot_heights[:, R_index] < self.height_clip, 1, 0)
        center_count_RR_foot = torch.where(self.measured_RR_foot_heights[:, R_index] < self.height_clip, 1, 0)

        count_FL_foot = torch.sum(center_count_FL_foot, dim=-1, keepdim=True)
        count_FR_foot = torch.sum(center_count_FR_foot, dim=-1, keepdim=True)
        count_RL_foot = torch.sum(center_count_RL_foot, dim=-1, keepdim=True)
        count_RR_foot = torch.sum(center_count_RR_foot, dim=-1, keepdim=True)

        count_FL_foot = torch.where(count_FL_foot > 0, 1, 0)
        count_FR_foot = torch.where(count_FR_foot > 0, 1, 0)
        count_RL_foot = torch.where(count_RL_foot > 0, 1, 0)
        count_RR_foot = torch.where(count_RR_foot > 0, 1, 0)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot,
                                       count_RL_foot, count_RR_foot), dim=-1)



        reward_feet_center = (self.terrain_levels > 3) * torch.sum(foot_center_count * contact_filt, dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        # print('contact_filt', contact_filt)
        # print('self.reward_feet_center', self.contact_filt)
        # print()
        return reward_feet_center


    def _reward_feet_center(self):

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)

        F_index = [1, 3, 5, 7]
        R_index = [1, 3, 5, 7]

        center_count_FL_foot = torch.where(self.measured_FL_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_FR_foot = torch.where(self.measured_FR_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_RL_foot = torch.where(self.measured_RL_foot_heights[:, R_index] < self.height_clip, 1, 0)
        center_count_RR_foot = torch.where(self.measured_RR_foot_heights[:, R_index] < self.height_clip, 1, 0)

        count_FL_foot = torch.sum(center_count_FL_foot, dim=-1, keepdim=True)
        count_FR_foot = torch.sum(center_count_FR_foot, dim=-1, keepdim=True)
        count_RL_foot = torch.sum(center_count_RL_foot, dim=-1, keepdim=True)
        count_RR_foot = torch.sum(center_count_RR_foot, dim=-1, keepdim=True)

        count_FL_foot = torch.where(count_FL_foot > 0, 1, 0)
        count_FR_foot = torch.where(count_FR_foot > 0, 1, 0)
        count_RL_foot = torch.where(count_RL_foot > 0, 1, 0)
        count_RR_foot = torch.where(count_RR_foot > 0, 1, 0)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot, count_RL_foot, count_RR_foot), dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        # print('contact_filt', contact_filt)

        # reward_feet_center1 = (self.terrain_levels > 3) * torch.sum(foot_center_count * contact_filt, dim=-1)
        reward_feet_center1 =  torch.sum(foot_center_count * contact_filt, dim=-1)

        F_index = [0, 2, 6, 8]
        R_index = [0, 2, 6, 8]

        center_count_FL_foot = torch.where(self.measured_FL_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_FR_foot = torch.where(self.measured_FR_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_RL_foot = torch.where(self.measured_RL_foot_heights[:, R_index] < self.height_clip, 1, 0)
        center_count_RR_foot = torch.where(self.measured_RR_foot_heights[:, R_index] < self.height_clip, 1, 0)

        count_FL_foot = torch.sum(center_count_FL_foot, dim=-1, keepdim=True)
        count_FR_foot = torch.sum(center_count_FR_foot, dim=-1, keepdim=True)
        count_RL_foot = torch.sum(center_count_RL_foot, dim=-1, keepdim=True)
        count_RR_foot = torch.sum(center_count_RR_foot, dim=-1, keepdim=True)

        count_FL_foot = torch.where(count_FL_foot > 0, 1, 0)
        count_FR_foot = torch.where(count_FR_foot > 0, 1, 0)
        count_RL_foot = torch.where(count_RL_foot > 0, 1, 0)
        count_RR_foot = torch.where(count_RR_foot > 0, 1, 0)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot, count_RL_foot, count_RR_foot), dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        # print('contact_filt', contact_filt)
        reward_feet_center2 = (self.terrain_levels > 3) * torch.sum(foot_center_count * contact_filt, dim=-1)
        # print('self.contact_filt', self.contact_filt)
        # print('reward_feet_center', reward_feet_center, reward_feet_center.shape)
        reward_feet_center = reward_feet_center1 + 2*reward_feet_center2
        return reward_feet_center

    def _reward_feet_second_center(self):

        F_index = [0, 2, 6, 8]
        R_index = [0, 2, 6, 8]

        center_count_FL_foot = torch.where(self.measured_FL_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_FR_foot = torch.where(self.measured_FR_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_RL_foot = torch.where(self.measured_RL_foot_heights[:, R_index] < self.height_clip, 1, 0)
        center_count_RR_foot = torch.where(self.measured_RR_foot_heights[:, R_index] < self.height_clip, 1, 0)

        count_FL_foot = torch.sum(center_count_FL_foot, dim=-1, keepdim=True)
        count_FR_foot = torch.sum(center_count_FR_foot, dim=-1, keepdim=True)
        count_RL_foot = torch.sum(center_count_RL_foot, dim=-1, keepdim=True)
        count_RR_foot = torch.sum(center_count_RR_foot, dim=-1, keepdim=True)

        count_FL_foot = torch.where(count_FL_foot > 0, 1, 0)
        count_FR_foot = torch.where(count_FR_foot > 0, 1, 0)
        count_RL_foot = torch.where(count_RL_foot > 0, 1, 0)
        count_RR_foot = torch.where(count_RR_foot > 0, 1, 0)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot,
                                       count_RL_foot, count_RR_foot), dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        # print('contact_filt', self.contact_filt)
        # print()
        reward_feet_center = (self.terrain_levels > 3) * torch.sum(foot_center_count * self.contact_filt, dim=-1)
        # print('self.contact_filt', self.contact_filt)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot,
                                       count_RL_foot, count_RR_foot), dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        reward_feet_center = (self.terrain_levels > 3) * torch.sum(foot_center_count * self.contact_filt, dim=-1)
        # print('self.contact_filt', self.contact_filt)
        # print('reward_feet_center', reward_feet_center, reward_feet_center.shape)
        return reward_feet_center

    def _reward_feet_third_center(self):

        F_index = [8, 9, 10, 14, 18, 21, 25, 28, 32, 36, 37, 38]
        R_index = [22, 23, 24, 28, 32, 35, 39, 42, 46, 50, 51, 52]

        center_count_FL_foot = torch.where(self.measured_FL_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_FR_foot = torch.where(self.measured_FR_foot_heights[:, F_index] < self.height_clip, 1, 0)
        center_count_RL_foot = torch.where(self.measured_RL_foot_heights[:, R_index] < self.height_clip, 1, 0)
        center_count_RR_foot = torch.where(self.measured_RR_foot_heights[:, R_index] < self.height_clip, 1, 0)

        count_FL_foot = torch.sum(center_count_FL_foot, dim=-1, keepdim=True)
        count_FR_foot = torch.sum(center_count_FR_foot, dim=-1, keepdim=True)
        count_RL_foot = torch.sum(center_count_RL_foot, dim=-1, keepdim=True)
        count_RR_foot = torch.sum(center_count_RR_foot, dim=-1, keepdim=True)

        # print('1', center_count_FL_foot, count_FL_foot)
        # print('2', center_count_FR_foot, count_FR_foot)
        # print('3', center_count_RL_foot, count_RL_foot)
        # print('4', center_count_RR_foot, count_RR_foot)

        foot_center_count = torch.cat((count_FL_foot, count_FR_foot,
                                       count_RL_foot, count_RR_foot), dim=-1)

        # print('foot_center_count', foot_center_count, foot_center_count.shape)
        reward_feet_center = torch.sum(foot_center_count * self.contact_filt, dim=-1)
        # print('self.contact_filt', self.contact_filt)
        # print('reward_feet_center', reward_feet_center, reward_feet_center.shape)
        return reward_feet_center

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

        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        rew[~self.step_gap_ids] *= 0

        return rew

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                        4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        rew = torch.square(self.base_lin_vel[:, 2])
        # print(self.env_class)
        if self.cfg.terrain.mesh_type in ["mix"]:
            if len(self.step_gap_ids) > 0:
                rew[self.env_class == 5] *= 0.25
                rew[self.env_class == 6] *= 1
            pass

        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            if len(self.step_gap_ids) > 0:
                rew[self.env_class == 12] *= 0.25
                rew[self.env_class == 1] *= 1
            pass

        return rew

    def _reward_orientation(self):
        # Penalize non flat base orientation
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        if self.cfg.terrain.mesh_type in ["mix"]:
            if len(self.step_gap_ids) > 0:
                rew[self.env_class == 5] *= 0.5
                rew[self.env_class == 6] *= 10
            pass
        if self.cfg.terrain.mesh_type in ["gap_parkour"]:
            if len(self.step_gap_ids) > 0:
                rew[self.env_class == 12] *= 0.5
                rew[self.env_class == 1] *= 10
            pass
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
        return rew_airTime

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_yaw(self):
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.terrain.mesh_type in ["mix"]:
            rew[self.env_class == 5] = torch.abs(self.base_ang_vel[self.env_class == 5, 2])
            rew[self.env_class == 6] = torch.abs(self.base_ang_vel[self.env_class == 6, 2])
        return rew

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        return rew

