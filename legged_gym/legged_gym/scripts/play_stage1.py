from operator import index

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, update_cfg_from_args
import pandas as pd
import numpy as np
import torch

from termcolor import cprint
import torch.nn.functional as F

def play(args, EXPORT_POLICY, MOVE_CAMERA, RECORD_FRAMES, LOG_STATES=False):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    print('env_cfg, train_cfg',  env_cfg, train_cfg)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 20)
    if getattr(env_cfg, "camera", None) is not None:
        if getattr(args, 'update_wm', None) is not None:
            env_cfg.camera.update_wm = args.update_wm
        if getattr(args, 'load_world_model_path', None) is not None:
            env_cfg.camera.load_world_model_policy_file = args.load_world_model_path
        if getattr(args, 'compare_depth_vis', None) is not None:
            env_cfg.camera.render_compare_pre_vis = args.compare_depth_vis
            if args.compare_depth_vis:
                vis_root = args.load_world_model_path or args.resume_name or args.output_name
                if vis_root:
                    env_cfg.camera.compare_vis_save_dir = os.path.join(vis_root, 'vis_depth')
                    cprint(
                        f"Depth compare vis ON -> OpenCV window + {env_cfg.camera.compare_vis_save_dir}/latest.png",
                        'cyan', attrs=['bold'],
                    )
        env_cfg.camera.load_world_model_policy = True

    env_cfg.terrain.num_cols_half = env_cfg.terrain.num_cols
    env_cfg.terrain.border_size = 5  # [m]
    env_cfg.terrain.max_init_terrain_level = 6  # starting curriculum state

    env_cfg.terrain.num_rows = env_cfg.terrain.max_init_terrain_level + 1
    env_cfg.terrain.num_cols = 3

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False

    # fixed velocity direction evaluation (make sure the value is within the training range)
    # env_cfg.commands.ranges.mix_lin_vel_x = [2.9, 2.9]
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0.7]
    env_cfg.commands.ranges.new_lin_vel_x =[0.7, 0.7]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading =  [0.0, 0.0]

    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    if getattr(args, 'vis_env_id', None) is not None:
        cprint(
            f"Visualization ref_env={env_cfg.viewer.ref_env} (num_envs={env_cfg.env.num_envs})",
            'cyan', attrs=['bold'],
        )

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs_dict = env.reset()
    # load policy
    train_cfg.runner.resume = True # set the mode to be evalution
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)



    policy, pre_extrin_en = ppo_runner.get_inference_policy(device=env.device)

    logger = Logger(env.dt)
    robot_index = getattr(env_cfg.viewer, 'ref_env', 0)
    stop_state_log = 300 if LOG_STATES else 0
    stop_rew_log = env.max_episode_length + 1 if LOG_STATES else 0
    cam_offset = (
        np.array(env_cfg.viewer.pos, dtype=np.float64)
        - np.array(env_cfg.viewer.lookat, dtype=np.float64)
    )
    look_ahead = np.array([1.5, 0.0, 0.0])
    img_idx = 0

    real_vels = []; real_contacts = []; pre_vels=[]; pre_contacts = []; steps = []
    for i in range(1000*int(env.max_episode_length)):

        actions = policy(obs_dict)

        extrin_en = pre_extrin_en(obs_dict)[1][0]


        obs_dict, rews, dones, infos = env.step(actions.detach())

        # images_clean = obs_dict['image_buf'][0, 1].detach().cpu().numpy()
        # print('images_clean', images_clean)
        # print()
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            robot_pos = env.root_states[robot_index, :3].detach().cpu().numpy()
            camera_direction = robot_pos + look_ahead
            camera_position = camera_direction + cam_offset
            env.set_camera(camera_position, camera_direction)
        steps.append(i)
        # print('ds', i, extrin_en)
        if i < stop_state_log:
            logger.log_states(
                {
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),

                    'real_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'real_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'real_vel_z': env.base_lin_vel[robot_index, 2].item(),

                    'pre_vel_x': extrin_en[robot_index, 0:1].item()/2,
                    'pre_vel_y': extrin_en[robot_index, 1:2].item()/2,
                    'pre_vel_z': extrin_en[robot_index, 2:3].item()/2,

                    'real_yaw': env.base_ang_vel[robot_index, 2].item(),

                }
            )

        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, EXPORT_POLICY, MOVE_CAMERA, RECORD_FRAMES)
    
