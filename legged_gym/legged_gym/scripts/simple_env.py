from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)

    env_cfg.terrain.max_init_terrain_level = 9  # starting curriculum state
    env_cfg.terrain.num_rows = env_cfg.terrain.max_init_terrain_level+1
    env_cfg.terrain.num_cols = 13
    # env_cfg.terrain.border_size = 5  # [m]

    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    # TODO: hand-crafted policy
    for i in range(1000 * int(env.max_episode_length)):
        actions = torch.zeros(env.num_actions, dtype=torch.float)
        actions = actions.repeat(env_cfg.env.num_envs, 1)
        obs_dict, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    play(args)