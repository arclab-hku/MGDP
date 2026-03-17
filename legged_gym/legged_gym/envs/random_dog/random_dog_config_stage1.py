from legged_gym.envs.baseline.legged_robot_config_baseline import LeggedRobotBaseCfg, LeggedRobotBaseCfgPPO
import numpy as np
import os


class RandomCfgStage1(LeggedRobotBaseCfg):
    encoder_ppo_ref = None

    class env(LeggedRobotBaseCfg.env):
        num_privileged_obs = 3 + 4 + 17 * 11

        train_type = "MGDP"
        num_env_morph_priv_obs = 33
        num_histroy_obs = 4
        graphics_device_num = 0
        use_history = True
        env_gait = 3

    class camera(LeggedRobotBaseCfg.camera):
        render_compare_pre_vis = False
        render_compare_pre_map = False
        camera_type = 'warp'
        render_compare_pre_map_real_data = False
        debug_height_stats = False
        use_camera = True
        world_model = True
        use_memory = True

        noise_gaussian = 0.03
        noise_dropout = 0.1

        normalize = True
        resized = [16, 16]
        near_clip = 0.0
        far_clip = 5.0

        update_wm = True
        use_map_decoder = True

        wm_visual_mse_weight = 1.2
        wm_visual_l1_weight = 0.0
        wm_height_mse_weight = 1.0
        wm_height_l1_weight = 0.0
        wm_height_visual_mse_weight = 0.0
        wm_contrastive_weight = 0.3
        wm_best_visual_weight = 0.6
        wm_best_height_weight = 0.4

        wm_lr = 1e-4
        wm_decoder_lr_scale = 1.0
        wm_weight_decay = 1e-5
        wm_grad_clip = 1.0

        update_interval = 5

        load_world_model_policy = False
        load_world_model_policy_file = "{LEGGED_GYM_ROOT_DIR}" + '/models/MGDP/stage1/001'

    class terrain(LeggedRobotBaseCfg.terrain):
        mesh_type = 'mix'
        measure_heights = True
        terrain_dict = {
            "slope down": 0.2,
            "pyramid": 0.2,
            "stairs down": 0.2,
            "stairs up": 0.2,
            "discrete obstacles": 1.1,
            "hurdle": 0.2,
            "gap": 1.2,
            "ramp": 1.1,
            "bream": 0.0,
            "new stairs down": 0.3,
            "pit": 1,
        }
        terrain_proportions = list(terrain_dict.values())

        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 10
        edge_width_thresh = 0.05
        simplify_grid = True


        x_min, x_max, x_step = -0.40, 1.30, 0.1
        y_min, y_max, y_step = -0.50, 0.60, 0.1
        measured_points_x = np.round(np.arange(x_min, x_max, x_step), 2)
        measured_points_y = np.round(np.arange(y_min, y_max, y_step), 2)
        print(measured_points_x, measured_points_y)

        num_point_x = measured_points_x.shape[0]
        num_point_y = measured_points_y.shape[0]

    class commands(LeggedRobotBaseCfg.commands):
        curriculum = True
        heading_command = True
        zero_command = True
        new_max_curriculum = 1.0
        new_min_curriculum = 0.0

        class ranges(LeggedRobotBaseCfg.commands.ranges):
            new_lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            new_lin_vel_y = [-1.0, 1.0]  
            new_ang_vel_yaw = [-1.0, 1.0]  
            new_heading = [-3.14, 3.14]

            # lin_vel_x = [0.0, 1.5]
            # lin_vel_y = [0.0, 0.0]
            # ang_vel_yaw = [0, 0]
            # heading = [0, 0]

    class init_state(LeggedRobotBaseCfg.init_state):
        pos = [0, 0, 0.45]
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
        dog_names = os.environ.get("DOG_NAMES", "")
        dog_name = os.environ.get("DOG_NAME", "go2")
        asset_name = (
            [n.strip() for n in dog_names.split(",") if n.strip()]
            if dog_names else [dog_name]
        )
        penalize_contacts_on_narrow = ["base", "Head"]
        terminate_after_contacts_on_narrow = ["base"]

        penalize_contacts_on = ["base", "thigh", "calf", "Head"]
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
        terrain_adaptive_reward = True

        class scales(LeggedRobotBaseCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1
            ang_vel_xy = -0.05
            torques = -1e-5
            dof_acc = -2.5e-7
            action_rate = -0.01
            orientation = -0.2
            collision = -1

            motion_trot = -0.1

            feet_air_time = 1
            feet_stumble = -1

            stand_still = -0.1


class RandomCfgPPOStage1(LeggedRobotBaseCfgPPO):
    class runner(LeggedRobotBaseCfgPPO.runner):
        run_name = ''
        max_iterations = 60001
        resume = False
        save_interval = 2000
        experiment_name = 'random_dog_stage1'

    class Encoder(LeggedRobotBaseCfgPPO.Encoder):
        checkpoint_model = None
        camera_dim = RandomCfgStage1.camera.resized
        HistoryLen = 4
        output_dims = 32
        encoder_output = 7
        pool = 1
        MLPModule_info = {
            'input_dims': 45 * HistoryLen,
            'hidden_dims': [256, 128, encoder_output],
        }

        CNNModule_info = {
            'input_channels': 2,
            'hidden_channels': [32, 64, 64],
            'output_channels': 64,
            'pool': 2
        }

        MapModule_info = {
            "input_channels": 1,
            "hidden_channels": [16, 32, 64],
            "output_channels": 32,
            "pool": 2
        }

        GRUModule_info = {
            'input_dims': output_dims * 2,
            'rnn_type': 'gru',
            'rnn_num_layers': 1,
            'rnn_hidden_dims': output_dims * 2,

        }
        height_encoder_type = "CNN"
        cnn_mlp_units = [16]
        DecoderModule_info = [64, 128, 256]


RandomCfgStage1.encoder_ppo_ref = RandomCfgPPOStage1
