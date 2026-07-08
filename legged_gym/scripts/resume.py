import os

# Available dog names: a1, aliengo, anymal_c, b1, go1, go2, lite3, mini_cheetah, mini_point, solo, spot
# DOG_NAMES = ["aliengo", "anymal_c", "b1", "go1", "lite3", "mini_cheetah", "mini_point",  "spot"]
DOG_NAMES = [ "go2"]

os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import *

CUDA_DEVICE_ID = 0

args = get_args()
args.task = 'random_dog_stage2'
args.num_envs = 64
args.headless = True
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID
args.seed = 1
args.algo = 'MGDP'
args.load_world_model_policy = True

args.checkpoint_model = 'last.pt'
args.resume = True

stage1_model_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs/random_dog/MGDP/stage1/baseline')
args.resume_name = stage1_model_dir
args.load_world_model_path = stage1_model_dir

args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs/random_dog/MGDP/stage2/resume')

print("args.output_name:", args.output_name)
train(args)
