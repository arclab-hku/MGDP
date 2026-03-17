from legged_gym.envs.baseline.legged_robot_config_baseline import LeggedRobotBaseCfg, LeggedRobotBaseCfgPPO
from legged_gym.envs.random_dog.random_dog_config_stage1 import RandomCfgPPOStage1
import numpy as np


class RandomCfgStage2(LeggedRobotBaseCfg):
    encoder_ppo_ref = None

    class env(LeggedRobotBaseCfg.env):
        num_privileged_obs = 3 + 4 + 17 * 11
        train_type = "MGDP"
        num_env_morph_priv_obs = 33
        num_histroy_obs = 4
        graphics_device_num = 0
        use_history = True
        env_gait = 1

    class camera(LeggedRobotBaseCfg.camera):
        render_compare_pre_vis = False
        render_compare_pre_map = False
        camera_type = 'warp'
        render_compare_pre_map_real_data = False
        use_camera = True
        world_model = True
        use_memory = True

        # ####### without ########
        # noise_gaussian = None
        # noise_dropout =  None

        # ###### with  noise_gaussian ########
        # noise_gaussian = 0.03
        # noise_dropout =  None
        #
        # ####### with noise_gaussian ########
        # noise_gaussian = None
        # noise_dropout =  0.04
        #
        # ####### with noise_gaussian_dropout ########
        noise_gaussian = 0.03
        noise_dropout = 0.1


        normalize = True

        update_wm = True

        use_map_decoder = True

        load_world_model_policy = True
        load_world_model_policy_file = "{LEGGED_GYM_ROOT_DIR}" +'/models/MGDP/stage1/001'

    class terrain(LeggedRobotBaseCfg.terrain):
        mesh_type = 'gap_parkour'
        measure_heights = True
        terrain_dict = {
            "plane": 0.0,
            "up_stairs": 0.0,
            "down_stairs": 0.0,
            "single-gap": 0.002,
            "step-stone": 0.101,
            "Stones-2Rows": 0.101,
            "balance-2Stones": 0.0,
            "stones-1Rows": 0.101,
            "single-bridge": 0.101,
            "step-Beams": 0.0,
            "Rotation-Beams": 0.0,
            "narrow-Beams": 0.0,
            "cross-Beams": 0.0,
            "air-Beams": 0.101,
            "air_stone": 0.101,
            "hurdle": 0.101,
            "ramp": 0.101,
            "corridor": 1.1,
        }
        curriculum = True
        terrain_proportions = list(terrain_dict.values())
        horizontal_scale = 0.05
        vertical_scale = 0.005
        simplify_grid = True
        edge_width_thresh = 0.05
        border_size = 5
        add_roughness = True
        height = [0.01, 0.04]
        downsampled_scale = 0.5
        terrain_length = 10.
        terrain_width = 4.
        num_goals = 10
        num_rows = 10
        num_cols = 10
        x_min, x_max, x_step = -0.40, 1.30, 0.1
        y_min, y_max, y_step = -0.50, 0.60, 0.1
        measured_points_x = np.round(np.arange(x_min, x_max, x_step), 2)
        measured_points_y = np.round(np.arange(y_min, y_max, y_step), 2)
        num_point_x = measured_points_x.shape[0]
        num_point_y = measured_points_y.shape[0]
        add_air_beam = True
        add_air_stone = True

    class commands(LeggedRobotBaseCfg.commands):
        curriculum = True
        heading_command = True
        zero_command = True
        new_max_curriculum = 1.0
        new_min_curriculum = 0.0

        class ranges(LeggedRobotBaseCfg.commands.ranges):
            new_lin_vel_x = [0.0, 1.5]
            new_lin_vel_y = [0.0, 0.0]
            new_ang_vel_yaw = [0, 0]
            new_heading = [0, 0]

            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

    class init_state(LeggedRobotBaseCfg.init_state):
        pos = [-4, 0, 0.45]  # x,y,z [m]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,

            'FR_hip_joint': -0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,

            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,

            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotBaseCfg.control):
        control_type = "P"
        stiffness = {'joint': 30.0}
        damping = {'joint': 0.8}

    class asset(LeggedRobotBaseCfg.asset):
        asset_name = ["go2"]
        penalize_contacts_on_narrow = ["base", "Head"]
        terminate_after_contacts_on_narrow = ["base"]

        penalize_contacts_on = ["base", "Head"]
        terminate_after_contacts_on = ["base"]

    class privInfo(LeggedRobotBaseCfg.privInfo):
        enableMeasuredVel = True
        enablePayload = False
        enableFriction = False
        enableStiffnessDamping = False
        enableMotorStrength = False
        enablemMotorOffsets = False
        enableCom = False
        enableLimb_mass = False
        enableForce = False
        enableFootContact = False
        enableFootHeight = True

        enableMaxFootHeight = False

        enableMeasuredHeight = True

    class domain_rand(LeggedRobotBaseCfg.domain_rand):
        randomize_action_latency = True
        latency_range = [0.00, 0.02]

    class rewards(LeggedRobotBaseCfg.rewards):
        terrain_adaptive_reward = False
        class scales(LeggedRobotBaseCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -1
            ang_vel_xy = -0.05
            torques = -1e-5
            dof_acc = -2.5e-7
            action_rate = -0.01
            orientation = -0.2
            collision = -1

            feet_air_time = 1
            feet_stumble = -1
            feet_edge = -1.0

            motion_trot = -0.1
            dof_pos_limits = -10.0
            torque_limits = -0.0


class RandomCfgPPOStage2(RandomCfgPPOStage1):
    class runner(RandomCfgPPOStage1.runner):
        experiment_name = 'random_dog_stage2'


RandomCfgStage2.encoder_ppo_ref = RandomCfgPPOStage2
