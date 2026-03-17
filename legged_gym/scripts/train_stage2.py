"""阶段一训练（World Model + 运动）。"""
import isaacgym
from legged_gym.scripts.train import *
import os

CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

args = get_args()
args.task = 'random_dog_stage2'
args.num_envs = 4
args.headless = False
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID

args.seed = 1
args.algo = 'MGDP'

args.load_world_model_policy = False

args.output_name = parent_dir + '/outputs/random_dog/MGDP/stage2/001'

print("args.output_name:", args.output_name)
train(args)
