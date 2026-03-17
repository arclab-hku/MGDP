# Warp-Sensor

## Installation
Fistly, you need to install the following packages:

```bash
cd warp_sensor && pip install -e .
```

## Test 

After installation, you can test the warp sensor with the following code:

```python
warp-cam # test the camera
```

and you will see the point cloud.

## Configuration

> **If you want to automatically fix the border bias of terrain, you should set the following configurations:**

```yaml
terrain_process:
  offset: True
```

Otherwise, please keep the default configuration or just remove the configuration in the situation that you make sure the terrain border bias is fixed.

## Usage

in Isaacgym, you need to initialize the warp sensor with the following code:

```python
from warp_sensor import WarpManager, Config as WarpConfig, Camera, Lidar

# here self is to transfer the the gym environment to the warp sensor
cfg = WarpConfig()
cfg.camera = Camera() # if one more camera
self.warp_manager = WarpManager(self.num_envs, self, cfg=cfg, device=self.device)
```

and get updated observation with the following code:

```python
self.warp_manager.warp_update_frame() 
img = self.warp_manager['camera'].data
```
