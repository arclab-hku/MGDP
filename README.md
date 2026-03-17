# MGDP: Mastering a Generalized Depth Perception Model for Quadruped Locomotion

Website: https://arclab-hku.github.io/MGDP/

Video: https://youtu.be/yOGQvbQMUKE



# Isaac Gym Environments for Legged Robots

This repository provides the environment used to train Unitree Go1, Aliengo and Arcdog to walk on rough terrain using NVIDIA's Isaac Gym. It includes all components needed for sim-to-real transfer: actuator network (TODO), friction & mass randomization, noisy observations and random pushes during training.

**Maintainer**: Dong Yinzhao  
**Affiliation**: HKU

---

### Installation

1. Create a new Python virtual env with Python 3.8 (3.8.20 recommended).
   - `conda create -n MGDP_1 python=3.8.20`
2. Install PyTorch 1.10 with CUDA 11.3:
   ```bash
   pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
3. Install Isaac Gym:
   - `cd MGDP`
   - `cd isaacgym/python && pip install -e .`
   - Try running an example: `cd examples && python 1080_balls_of_solitude.py`
4. Install this repo (e.g. `legged_gym` / NavEnvs):
   - Clone the repository.
   - `pip install -e .`
   - `pip install -r requirement-gpu.txt`
5. Install Warp:
   - `cd warp_sensor && pip install -e .`
   - Test: `warp-cam`
   - exit: `esc`
---

### Usage
1. **Train a Generalized Depth Perception Model**
   From repo root:
   ```bash
   cd legged_gym/scripts
   python train.py
   ```
   Edit `train.py` to set `args.task` (e.g. `random_dog_stage1`), `args.output_name`, GPU id, etc.

2. **Train a Generalized Perception-based Locomotion Controller**
   ```bash
   cd legged_gym/scripts
   python resume.py
   ```
   Edit `resume.py` to set `args.resume_name` (previous run path), `args.output_name` (save path), and `args.task` (e.g. `random_dog_stage2`).

   **Select robot(s)**
   - Edit `DOG_NAMES = [...]` to mix multiple dogs in one run (envs use `dog_id = i % len(DOG_NAMES)`).


3. **Play / visualize**
   ```bash
   cd legged_gym/scripts
   python vis_stage1.py
   # or
   python vis_stage2.py
   ```
   - `vis_stage1.py`: to visualize the Generalized Depth Perception Model.
  ```bash
   python legged_gym/scripts/vis_stage1.py
   ```

   - `vis_stage2.py`: to visualize the Generalized Perception-based Locomotion Controller.
  ```bash
   python legged_gym/scripts/vis_stage2.py
   ```

4. **View the terrain**
   ```bash
   python legged_gym/scripts/play_terrain.py
   ```

