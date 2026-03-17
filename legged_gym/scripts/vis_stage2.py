import os

# Available dog names: a1, aliengo, anymal_c, b1, go1, go2, lite3, mini_cheetah, mini_point, solo, spot
# DOG_NAMES = ["aliengo", "anymal_c", "b1", "go1", "lite3", "mini_cheetah", "mini_point",  "spot"]
DOG_NAMES = [ "go2"]

os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)
from legged_gym.scripts.play_stage2 import *


CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = True
args = get_args()

args.task = 'random_dog_stage2'
args.num_envs = 64
args.headless = False
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID
args.checkpoint_model = 'model_26000.pt'
args.load_world_model_policy = True
args.update_wm = False

args.algo = 'MGDP'
real_path = "/models/MGDP/stage2/001"

args.output_name = parent_dir + real_path
args.resume_name = parent_dir + real_path
args.load_world_model_path = parent_dir + real_path
play(args, EXPORT_POLICY, RECORD_FRAMES, MOVE_CAMERA)
