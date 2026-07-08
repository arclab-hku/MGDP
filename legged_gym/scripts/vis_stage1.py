import os

DOG_NAMES = [ "go2"]

os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play_stage1 import *

CUDA_DEVICE_ID = 0

EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False       # follow ref_env robot; set False for a fixed overview camera
LOG_STATES = False       # set True to pop up velocity comparison plots after ~300 steps
COMPARE_DEPTH_VIS = True # noisy / predicted / clean depth panel (Stage 1 main visual)
COMPARE_HEIGHT_VIS = True # predicted / real sparse elevation-map panel
VIS_ENV_ID = 0           # default env to visualize (0-based); override via --vis_env_id

args = get_args()

args.task = 'random_dog_stage1'
args.num_envs = 4
args.headless = False

cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = CUDA_DEVICE_ID
args.checkpoint_model = 'last.pt'
args.load_world_model_policy = True
args.update_wm = False
if args.compare_depth_vis is None:
    args.compare_depth_vis = COMPARE_DEPTH_VIS
if args.compare_height_vis is None:
    args.compare_height_vis = COMPARE_HEIGHT_VIS
if args.vis_env_id is None:
    args.vis_env_id = VIS_ENV_ID

args.algo = 'MGDP'
model_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'models/MGDP/stage1/baseline')

args.output_name = model_dir
args.resume_name = model_dir
args.load_world_model_path = model_dir
play(args, EXPORT_POLICY, MOVE_CAMERA, RECORD_FRAMES, LOG_STATES)
