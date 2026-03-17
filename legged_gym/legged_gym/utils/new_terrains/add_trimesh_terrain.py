from isaacgym import terrain_utils
def trimesh_terrain(terrain, choice, slope,
                    proportions, step_height, discrete_obstacles_height, stepping_stones_size,
                    stone_distance, gap_size, pit_depth, ):
    if choice < proportions[0]:
        if choice < proportions[0] / 2:
            slope *= -1
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
    elif choice < proportions[1]:
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                             downsampled_scale=0.2)
    elif choice < proportions[3]:
        if choice < proportions[2]:
            step_height *= -1
        terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)

    elif choice < proportions[4]:
        num_rectangles = 20
        rectangle_min_size = 1.
        rectangle_max_size = 2.
        terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                 rectangle_max_size, num_rectangles, platform_size=3.)
    elif choice < proportions[5]:
        terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                              stone_distance=stone_distance, max_height=0., platform_size=4.)
    elif choice < proportions[6]:
        gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
    else:
        pit_terrain(terrain, depth=pit_depth, platform_size=4.)



def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth




