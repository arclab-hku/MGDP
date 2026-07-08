from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
class LeggedRobotBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 45
        num_privileged_obs = 0
        train_type = "Moral"
        num_env_priv_obs = 17
        num_env_morph_priv_obs = 19
        num_histroy_obs = 1

    class noise(LeggedRobotCfg.noise):
        add_privileged_noise = False
        add_height_noise = True

    class camera(LeggedRobotCfg.camera):
        use_camera = False
        camera_res = [16, 16]
        image_nums = 2
        near_clip = 0
        far_clip = 5
        resized = [16, 16]
        camera_type = "d"
        trans = (0.35, 0.0, 0.0)
        rot1 = (0.0, np.deg2rad(45), 0.0)
        hight_point_type = "min_pooling"
        depth_nosie = 0.1
        horizontal_fov = 87
        update_interval = 5

        loc_cam = False
        use_lidar = False

        world_model = False
        load_world_model_policy = False
        use_memory = False

    class env_init_info(LeggedRobotCfg.env_init_info):
        feet_height = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        x_min, x_max, x_step = -0.40, 1.30, 0.1
        y_min, y_max, y_step = -0.50, 0.60, 0.1
        measured_points_x = np.round(np.arange(x_min, x_max, x_step), 2)
        measured_points_y = np.round(np.arange(y_min, y_max, y_step), 2)

        num_point_x = measured_points_x.shape[0]
        num_point_y = measured_points_y.shape[0]

    class commands(LeggedRobotCfg.commands):
        curriculum = True

    class init_state(LeggedRobotCfg.init_state):
        pos = [-0.0, 0.0, 0.45]
        default_joint_angles = {
            "FL_hip_joint": -0.05,
            "RL_hip_joint": -0.05,
            "FR_hip_joint": 0.05,
            "RR_hip_joint": 0.05,

            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.8}
        action_scale = 0.25
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_aliengo_renet_aggre_it800.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/all_dog'
        name = "random_dog"
        asset_name = ["aliengo"]
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base", "trunk", "imu"]
        terminate_after_contacts_on = ["base", "trunk", "imu"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_friction = True
        added_friction_range = [0.2, 1.2]

        randomize_restitution = True
        restitution_range = [0, 1.0]

        randomize_com = True
        added_com_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = False
        added_lag_timesteps = 5
        added_lag_timesteps_sacle = [0, 3]

        randomize_motor_offset = True
        added_motor_offset = [-0.02, 0.02]

        randomize_action_latency = False
        latency_range = [0.00, 0.02]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.32
        max_contact_force = 100.0
        soft_dof_pos_limit = 0.9
        gait_threshold = [0.0, 1.2]
        lin_vel_clip = 0.1
        only_positive_rewards = True
        foot_height_target = 1.0
        gait_thod = None
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -1e-5
            dof_acc = -2.5e-7
            base_height = -0.0
            collision = -1.0
            action_rate = -0.01
            feet_air_time = 1.0
            stand_still = -0.0
            termination = -0.0

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, -15, 15]
        lookat = [10., 10, 5.]

    class evals(LeggedRobotCfg.evals):
        feet_stumble = True
        feet_step = True
        crash_freq = True
        any_contacts = True

    class privInfo(LeggedRobotCfg.privInfo):
        enableMeasuredVel = False
        enableMeasuredHeight = False
        enableForce = False
        enablePayload = False
        enableFriction = False
        enableRestitution = False
        enableCom = False
        enableStiffnessDamping = False
        enableMotorStrength = False
        enablemMotorOffsets = False
        enableMaxFootHeight = False
        enableEstTrajectory = False

class LeggedRobotBaseCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 2000
        resume = False
        save_interval = 500
        experiment_name = 'random_dog'
        export_policy = False

    class Encoder(LeggedRobotCfgPPO.Encoder):
        encoder_mlp_units = [256, 128, 7]
        priv_info_dim = 219
        checkpoint_model = None
        HistoryLen = 1
        Hist_info_dim = 45 * HistoryLen
        camera_dim = [16, 16]
        encoder_type = "History_MLP"
