from legged_gym.scripts.simple_env import *
import os

CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
print("current_dir:", current_dir)
parent_dir = os.path.dirname(current_dir) 
print("parent_dir:", parent_dir)

EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False
args = get_args()
args.load_world_model_policy = False

args.task = 'random_dog_stage2'  # random_dog_stage1, random_dog_stage2
args.num_envs = 4
args.headless = False
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda      
args.render_device = cuda  
args.sim_device = cuda     
args.graphics_device_num = CUDA_DEVICE_ID 


play(args)