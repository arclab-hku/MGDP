import isaacgym
from legged_gym.scripts.train import *
import os

CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
print("current_dir:", current_dir)
parent_dir = os.path.dirname(current_dir)
print("parent_dir2:", parent_dir)

args = get_args()
args.task = 'random_dog_stage2'
args.num_envs = 4096
args.headless = True
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID
args.seed = 1
args.algo = 'MGDP'
args.load_world_model_policy = True

args.checkpoint_model = 'model_10000.pt'
args.resume = True

real_path = '/models/MGDP/stage1/001'
args.resume_name = parent_dir + real_path
args.load_world_model_path = parent_dir + real_path

args.output_name = parent_dir + '/outputs/random_dog/MGDP/stage2/001'


train(args)
