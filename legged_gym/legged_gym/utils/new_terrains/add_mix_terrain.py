from isaacgym import terrain_utils
import numpy as np
import random
def trimesh_terrain(terrain, choice, difficulty, slope,
                    proportions, step_height, discrete_obstacles_height, stepping_stones_size,
                    stone_distance, gap_size, pit_depth, add_roughness, num_rows):
    depth = random.uniform(0.5, 0.5)
    # slope += 0.1

    if choice < proportions[0]:
        if choice < proportions[0] / 2:
            slope *= -1
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        idx = 0
    elif choice < proportions[1]:
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                             downsampled_scale=0.2)
        idx = 1
    elif choice < proportions[3]:
        if choice < proportions[2]:
            step_height *= -1
            idx = 2
        else:
            idx = 3

        terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)

    elif choice < proportions[4]:
        num_rectangles = 20
        rectangle_min_size = 1.
        rectangle_max_size = 2.5
        terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                 rectangle_max_size, num_rectangles, platform_size=3.)
        idx = 4

    elif choice < proportions[5]:

        parkour_step_terrain(terrain,
                             num_stones=1,
                             difficulty = difficulty,
                             x_range=[1.6, 2.0],
                             )
        add_roughness(terrain)
        idx = 5
    elif choice < proportions[6]:
        gap_size = 0.5 * difficulty if difficulty < 0.1 else 0.1 + difficulty / terrain.horizontal_scale
        parkour_step_gap_terrain(terrain, gap_size, depth=depth, platform_size=2)
        idx = 6
        add_roughness(terrain)
    elif choice < proportions[7]:
        half_sloped_terrain(terrain, level_index=difficulty * 7)
        idx = 7
        add_roughness(terrain)
    elif choice < proportions[8]:
        bream_length = 1 if difficulty < 0.2 else -0.4 * difficulty + 0.9
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        stepping_breams_terrain(
            terrain,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        idx = 8
        add_roughness(terrain)

    elif choice < proportions[9]:


        idx = 9
        new_step_height = step_height+0.2
        terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.5, step_height=new_step_height, platform_size=3.)
    else:
        pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        idx = 20

    terrain.idx = idx


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def parkour_step_gap_terrain(terrain, gap_size, depth,  platform_size=2.):
    gap_size = np.clip(int(gap_size), 1, 13)


    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    start_y = 0
    end_y = int(terrain.length-platform_size/8)

    start_x = platform_size
    center_x = terrain.width
    terrain.height_field_raw[start_x: center_x, start_y: end_y] = -depth
    terrain.height_field_raw[start_x+gap_size: center_x - gap_size, start_y+gap_size: end_y-gap_size] = 0

    # terrain.height_field_raw[:, :] = -depth


def half_sloped_terrain(terrain, level_index,   platform_size=2.):
    terrain_length = terrain.length
    slope_start = 5
    platform_size = int(platform_size / terrain.horizontal_scale)

    slope_end = int((terrain.width - platform_size) /2)

    height2width_ratio = 2*int(level_index+1)
    xs = np.arange(slope_start, slope_end)
    max_height_int = height2width_ratio*(slope_end - slope_start)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:slope_end, :] = heights[:, None]

    # print('terrain_length', slope_start, slope_end,  heights)
    # max 37.5, min 5.5

    x1 =  slope_end
    x2 =  int((terrain.width+ platform_size) /2)
    y1 = 0
    y2 = int(terrain.length )
    terrain.height_field_raw[x1:x2, y1:y2] = max_height_int

    slope_start = x2
    slope_end = terrain_length-5
    height2width_ratio = 2*int(level_index+1)
    xs = np.arange(slope_start, slope_end)
    max_height_int = height2width_ratio*(slope_end - slope_start)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    
    height_size = len(heights)

    
    reversed_index = height_size - 1

    for i in range(slope_start, slope_end):
        terrain.height_field_raw[i, :] = heights[reversed_index][None]
        reversed_index -= 1


def parkour_hurdle_terrain(terrain, platform_len=2.5, num_stones=8, stone_len=0.3,x_range = [1.5, 2.4],
                           y_range = [-0.4, 0.4], half_valid_width = [0.4, 0.8],hurdle_height = 0.2, platform_size=2):
    platform_size = int(platform_size / terrain.horizontal_scale)

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)

    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x = platform_len

    # print('hurdle_height_max', hurdle_height_max)

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = dis_y_max # np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, ] = hurdle_height_max
        terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2,   :2] = 0
        terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2,   mid_y*2-2:] = 0

        # goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    # final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

    # if final_dis_x > terrain.width:
    #     final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    # goals[-1] = [final_dis_x, mid_y]

    # terrain.goals = goals * terrain.horizontal_scale

    x1 =  int((terrain.width- platform_size) /2)
    x2 =  int((terrain.width+ platform_size) /2)
    y1 = 0
    y2 = int(terrain.length )
    terrain.height_field_raw[x1:x2, y1:y2] = 0


def parkour_step_terrain(terrain, num_stones=8, x_range=[0.2, 0.4], difficulty=0.1, platform_size=1.5):
    max_height = 1
    min_height = 0
    if difficulty < 0.1:
        hurdel_height_mix = difficulty + 0.04
        hurdel_height_max = hurdel_height_mix + 0.04
    elif 0.1 <= difficulty:
        hurdel_height_mix =  difficulty* (max_height - min_height) * 0.8
        hurdel_height_max = hurdel_height_mix + 0.01

    # print('difficulty', difficulty)
    
    platform_size = int(platform_size / terrain.horizontal_scale)
    dis_x_min = round((x_range[0]) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1]) / terrain.horizontal_scale)

    hurdle_height_max = round(hurdel_height_max / terrain.vertical_scale)
    hurdle_height_min = round(hurdel_height_mix / terrain.vertical_scale)
    step_height = np.random.randint(hurdle_height_min, hurdle_height_max)
    # print('hurdle', step_height, difficulty, hurdle_height_min, hurdle_height_max)
    
    max_x = int((terrain.width + platform_size) )

    
    start_y = 2
    end_y = int(terrain.length) - 2

    new_stair_height = 0

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)

        start_x = np.random.randint(60, 65)

        new_stair_height += step_height

        end_x = start_x + rand_x
        end_x = np.clip(end_x, start_x, max_x)
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = new_stair_height

    return terrain


def parkour_step_terrain1(terrain, num_stones=8, x_range=[0.2, 0.4], step_height=0.2, platform_size=1.5):
    

    
    platform_size = int(platform_size / terrain.horizontal_scale)
    dis_x_min = round((x_range[0]) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1]) / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)

    
    x1 = int((terrain.width - platform_size) / 2)
    x2 = int((terrain.width + platform_size) / 2)

    
    start_y = 2
    end_y = int(terrain.length) - 2

    
    middle_num = int(num_stones // 2)

    new_stair_height = 0
    end_x = 10
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)

        if i < middle_num:
            new_stair_height += step_height
            start_x = end_x
            end_x = end_x + rand_x
            end_x = np.clip(end_x, start_x, x2)
            terrain.height_field_raw[start_x:end_x, start_y:end_y] = new_stair_height

        elif i == middle_num:
            new_stair_height = step_height * middle_num
            start_x = x1
            end_x = x2
            terrain.height_field_raw[start_x:end_x, start_y:end_y] = new_stair_height

        elif i > middle_num:
            new_stair_height -= step_height
            start_x = end_x
            end_x = end_x + rand_x
            terrain.height_field_raw[start_x:end_x, start_y:end_y] = new_stair_height

    return terrain


def stepping_breams_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=1):

    
    bream_length = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    stone_distance = np.clip(stone_distance, 1, 5)

    platform_size = int(platform_size / terrain.horizontal_scale)
    max_height = np.clip(max_height, 0, 30)
    height_range = np.arange(0, max_height, step=4)

    
    platform_y = terrain.length // 2 - platform_size // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    min_bream_width = 15
    max_bream_width = 30

    
    x1 = terrain.width // 2 - platform_size // 2
    x2 = terrain.width // 2 + platform_size // 2

    
    start_x_front = terrain.width // 2 - platform_size // 2 -1
    while start_x_front >= 0:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
        stop_x_front = max(0, start_x_front - bream_length)
        terrain.height_field_raw[stop_x_front: start_x_front, row1_y: row1_y + bream_width] = np.random.choice(height_range)
        start_x_front -= bream_length + stone_distance

    
    start_x_back = terrain.width // 2 + platform_size // 2 +1
    while start_x_back < terrain.width:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
        stop_x_back = min(terrain.width, start_x_back + bream_length)
        terrain.height_field_raw[start_x_back: stop_x_back, row1_y: row1_y + bream_width] = np.random.choice(height_range)
        start_x_back += bream_length + stone_distance

    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0


