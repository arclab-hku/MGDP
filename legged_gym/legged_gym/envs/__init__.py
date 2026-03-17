
from .base.legged_robot import LeggedRobot

from .baseline.legged_robot_terrains import Legged_terrains
from .baseline.legged_robot_camera import Legged_camera, CameraMixin
from .baseline.legged_robot_rewards import Legged_rewards

from legged_gym.utils.task_registry import task_registry


from .random_dog.random_dog import Randomdog
from .random_dog.random_dog_config_stage1 import RandomCfgStage1, RandomCfgPPOStage1
from .random_dog.random_dog_config_stage2 import RandomCfgStage2, RandomCfgPPOStage2

# 第一阶段：采集数据 + 训练 world model（mix 地形等）
task_registry.register("random_dog_stage1", Randomdog, RandomCfgStage1(), RandomCfgPPOStage1())

# 第二阶段：与论文对齐（gap_parkour、reward 等），可加载 stage1 的 WM/策略继续训
task_registry.register("random_dog_stage2", Randomdog, RandomCfgStage2(), RandomCfgPPOStage2())


