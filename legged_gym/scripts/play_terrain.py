from legged_gym.scripts.simple_env import *
import os

# 可调参数：GPU 编号，改这里即可切换 cuda:0 / cuda:1 / ...
CUDA_DEVICE_ID = 0

current_dir = os.getcwd()
print("current_dir:", current_dir)
parent_dir = os.path.dirname(current_dir)  # 上一级目录
print("parent_dir:", parent_dir)

EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False
args = get_args()
args.load_world_model_policy = False

args.task = 'random_dog_stage1'  # random_dog_stage1, random_dog_stage2
args.num_envs = 2
args.headless = False
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda      # RL训练设备
args.render_device = cuda  # 渲染设备
args.sim_device = cuda      # ✅ 必须显式设置物理模拟设备
args.graphics_device_num = CUDA_DEVICE_ID  # 与上面保持一致，供 env.graphics_device_num 使用


play(args)