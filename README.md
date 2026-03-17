## MGDP: Mastering a Generalized Depth Perception Model for Quadruped Locomotion

<h3 align="center">
  <a href="https://arclab-hku.github.io/MGDP/">Project Website</a>
  | <a href="https://youtu.be/yOGQvbQMUKE">Youtube Video</a>
</h3>

<div align="center">
  <video width="80%" controls>
    <source src="static/videos/MGDP_1.mp4" type="video/mp4">
  </video>
</div>

## Abstract
Perception-based Deep Reinforcement Learning (DRL) controllers demonstrate impressive performance on challenging terrains. However, existing controllers still face core limitations, struggling to achieve both terrain generality and platform transferability, and are constrained by high computational overhead and sensitivity to sensor noise. To address these challenges fundamentally, we propose a generalized control framework: Mastering a Generalized Contrastive Depth Model (MGDP).
We leverage NVIDIA Warp to enable efficient parallel computation of depth images, thereby mitigating the inherent high computational cost. MGDP extracts low-dimensional terrain feature representations from multi-modal inputs (depth images and height maps) and integrates an explicit depth map denoising mechanism. This process not only facilitates effective decoupling of perception from dynamics but also significantly reduces the memory. Furthermore, we design terrain-adaptive reward functions that modulate penalty strengths according to terrain characteristics, enabling the policy to acquire complex locomotion skills (e.g., climbing, jumping, crawling, squeezing) in a single training stage without relying on distillation. Experimental results demonstrate that MGDP not only endows the policy with superior cross-terrain generalization capability but also enables fast and efficient fine-tuning across diverse quadruped robot morphologies via its pre-trained, dynamics-decoupled perception model. This vigorously advances the development of unified, efficient, and generalized frameworks for quadrupedal locomotion control.

## Installation
1. Create a Python virtual env with Python 3.8 (3.8.20 recommended).
   - `conda create -n MGDP_1 python=3.8.20`
2. Install PyTorch 1.10 with CUDA 11.3:
   ```bash
   pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
3. Install Isaac Gym:
   - `cd MGDP`
   - `cd isaacgym/python && pip install -e .`
   - Try running an example: `cd examples && python 1080_balls_of_solitude.py`
4. Install this repo:
   - Clone the repository
   - `pip install -e .`
   - `pip install -r requirement-gpu.txt`
5. Install Warp sensors:
   - `cd warp_sensor && pip install -e .`
   - Test: `warp-cam` (exit: `esc`)

## Usage
1. **Stage 1: Train a Generalized Depth Perception Model (MGDP Stage 1)**
   ```bash
   cd legged_gym/scripts
   python train.py
   ```
   - Edit `train.py` to set `args.task` (e.g. `random_dog_stage1`), `args.output_name`, GPU id, etc.

2. **Stage 2: Train a Generalized Perception-based Locomotion Controller (resume / fine-tune)**
   ```bash
   cd legged_gym/scripts
   python resume.py
   ```
   - Edit `resume.py` to set `args.resume_name` (previous run path), `args.output_name` (save path), and `args.task` (e.g. `random_dog_stage2`).
   - **Select robot(s)**:
     - Set `DOG_NAMES = [...]` to mix multiple dogs in one run (envs use `dog_id = i % len(DOG_NAMES)`).
     - If `DOG_NAMES` is not set, it falls back to `DOG_NAME`.

3. **Play / visualize**
   ```bash
   cd legged_gym/scripts
   python vis_stage1.py
   # or
   python vis_stage2.py
   ```
   - `vis_stage1.py`: visualize Stage 1 (Generalized Depth Perception Model / world model training).
   - `vis_stage2.py`: visualize Stage 2 (Generalized Perception-based Locomotion Controller).

4. **View the terrain**
   ```bash
   cd legged_gym/scripts
   python play_terrain.py
   ```

