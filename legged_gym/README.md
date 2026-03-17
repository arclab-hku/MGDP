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

1. **Train world model**  
   From repo root:
   ```bash
   cd legged_gym/scripts/
   python train.py
   ```
   Edit `train.py` to set `args.task` (e.g. `random_dog_stage1`), `args.output_name`, GPU id, etc.

2. **Resume / fine-tune locomotion**  
   ```bash
   cd scripts
   python resume.py
   ```
   Edit `resume.py` to set `args.resume_name` (path of the previous run), `args.output_name` (save path), and `args.task` (e.g. `random_dog_stage2`).

3. **Play a trained policy**  
   ```bash
   cd scripts
   python vis.py
   ```
   Or use `vis_joy.py` for joystick control; set `args.task` and policy path inside the script.

4. **View the terrain**  
   ```bash
   cd scripts
   python play_terrain.py
   ```

5. **Monitor GPU memory**  
   ```bash
   watch -n 0.1 nvidia-smi
   ```
