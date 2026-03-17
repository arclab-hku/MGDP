import os

# Available dog names: a1, aliengo, anymal_c, b1, go1, go2, lite3, mini_cheetah, mini_point, solo, spot
DOG_NAMES = [ "go2"]

os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

import isaacgym
from legged_gym.scripts.train import *

CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

args = get_args()
args.task = 'random_dog_stage1'
args.num_envs = 64
args.headless = True
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID

args.seed = 1
args.algo = 'MGDP'

args.load_world_model_policy = False

args.output_name = parent_dir + '/outputs/random_dog/MGDP/stage1/001'

print("args.output_name:", args.output_name)
train(args)
