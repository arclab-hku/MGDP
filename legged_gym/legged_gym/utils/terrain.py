import numpy as np
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.new_terrains import add_extreme_gap_terrain
from legged_gym.utils.new_terrains import add_trimesh_terrain
from legged_gym.utils.new_terrains import add_mix_terrain


import random
import pyfqmr
from scipy.ndimage import binary_dilation

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        if self.type == "mix":
            self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if self.type == "gap_parkour":
            self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
            self.num_goals = cfg.num_goals
            if self.num_goals != None:
                self.goals =  np.zeros((cfg.num_rows, 3))  # np.zeros((cfg.num_rows, cfg.num_cols, 3))
                self.goals_stone =  np.zeros((cfg.num_rows, 3))  # np.zeros((cfg.num_rows, cfg.num_cols, 3))
                self.goals_narrow =  np.zeros((cfg.num_rows, 3))  # np.zeros((cfg.num_rows, cfg.num_cols, 3))

                # self.center_position = np.zeros((cfg.num_rows, 3))

        if cfg.curriculum:
            # if self.type == "parkour":
            #     if cfg.max_difficulty == True:
            #         self.curiculum_parkour(random=True, max_difficulty=cfg.max_difficulty)
            #     else:
            # self.curiculum_parkour(random=True)
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        # self.heightsamples = self.height_field_raw

        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

        if self.type == "mix":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = add_mix_terrain.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold)

                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width * 2 + 1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count=int(0.05 * self.triangles.shape[0]), aggressiveness=7,
                                                  preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)

            else:
                self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                             self.cfg.horizontal_scale,
                                                                                             self.cfg.vertical_scale,
                                                                                             self.cfg.slope_treshold)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))


        if self.type == "gap_parkour":
            print("Converting heightmap to trimesh...")
            self.vertices, self.triangles, self.x_edge_mask = add_mix_terrain.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold)


            if cfg.add_air_beam:
                self.beam_vertices, self.beam_triangles = add_extreme_gap_terrain.get_muti_beam_trimeshes(
                    self.goals,
                    self.cfg.num_rows,
                    self.cfg.num_cols,
                    self.env_length,
                    self.env_width,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.border_size,
                )

                for i in range(len(self.beam_vertices)):
                    in_ver = self.beam_vertices[i][[0, 1, 4, 5], 0:3]
                    in_ver_x = [int(x / self.cfg.horizontal_scale) for x in in_ver[:, 0]]
                    in_ver_y = [int(x / self.cfg.horizontal_scale) for x in in_ver[:, 1]]
                    in_ver_z = in_ver[:, 2] / self.cfg.vertical_scale
                    self.height_field_raw[in_ver_x[0]: in_ver_x[1], in_ver_y[0]:in_ver_y[2]] = in_ver_z[0]
            else:
                self.beam_vertices, self.beam_triangles = [], []

            if cfg.add_air_stone:
                self.stone_vertices, self.stone_triangles = add_extreme_gap_terrain.get_muti_stone_trimeshes(
                    self.goals_stone,
                    self.cfg.num_rows,
                    self.cfg.num_cols,
                    self.env_length,
                    self.env_width,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.border_size,
                )

                for i in range(len(self.stone_vertices)):
                    in_ver = self.stone_vertices[i][[2, 3, 6, 7], 0:3]
                    in_ver_x = [int(x / self.cfg.horizontal_scale) for x in in_ver[:, 0]]
                    in_ver_y = [int(x / self.cfg.horizontal_scale) for x in in_ver[:, 1]]
                    in_ver_z = in_ver[:, 2] / self.cfg.vertical_scale
                    self.height_field_raw[in_ver_x[0]: in_ver_x[1], in_ver_y[0]:in_ver_y[2]] = in_ver_z[0]
            else:
                self.stone_vertices, self.stone_triangles = [], []

            half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
            structure = np.ones((half_edge_width * 2 + 1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

            if self.cfg.simplify_grid:
                mesh_simplifier = pyfqmr.Simplify()
                mesh_simplifier.setMesh(self.vertices, self.triangles)
                mesh_simplifier.simplify_mesh(target_count=int(0.05 * self.triangles.shape[0]), aggressiveness=7,
                                              preserve_border=True, verbose=10)

                self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                self.vertices = self.vertices.astype(np.float32)
                self.triangles = self.triangles.astype(np.uint32)


            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

        self.heightsamples = self.height_field_raw

    def randomized_terrain(self):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)

            # difficulty = 0.4
            difficulty = np.random.choice([0.2, 0.4, 0.6, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum_parkour(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)



    def selected_terrain(self):

        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.length_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.cfg.vertical_scale,
                                               horizontal_scale=self.cfg.horizontal_scale)

            # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            if i==0:
                terrain_type = self.cfg.terrain_kwargs.pop('type')
                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
                terrain.idx =  0
            elif i==1:
                terrain_type = self.cfg.terrain_kwargs1.pop('type')
                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs1)

                terrain.idx =  1
            else:
                terrain.idx =  2

            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        if self.cfg.add_roughness:
            max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
            height = random.uniform(self.cfg.height[0], max_height)
            terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)


    def make_terrain(self, choice, difficulty, obs_scale=1):
        terrain = terrain_utils.SubTerrain("terrains",
                                           width=self.length_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        if self.type == "gap_parkour":
            add_extreme_gap_terrain.trimesh_terrain(terrain, choice, difficulty, slope,
                                               self.proportions, step_height, discrete_obstacles_height,
                                               stepping_stones_size, stone_distance,
                                               gap_size, pit_depth, self.add_roughness, self.cfg.num_rows)

        elif self.type == "mix":
            add_mix_terrain.trimesh_terrain(terrain, choice, difficulty, slope,
                                                    self.proportions, step_height, discrete_obstacles_height,
                                                    stepping_stones_size, stone_distance,
                                                    gap_size, pit_depth, self.add_roughness, self.cfg.num_rows)

        elif self.type == "trimesh":
            add_trimesh_terrain.trimesh_terrain(terrain, choice, slope, self.proportions, step_height, discrete_obstacles_height,
                                                    stepping_stones_size, stone_distance,
                                                    gap_size, pit_depth)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        if self.type == "mix":
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            self.terrain_type[i, j] = terrain.idx

        if self.type == "gap_parkour":
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
            self.terrain_type[i, j] = terrain.idx


            if terrain.idx == 13 and self.num_goals != None:
                # print('9', i, terrain.center_position.shape)
                self.goals[i, 0:3] = terrain.center_position[i, 0:3] + [i * self.env_length, j * self.env_width, 0]
            if terrain.idx == 14 and self.num_goals != None:
                # print('14', i, terrain.center_position_stone.shape)
                self.goals_stone[i, 0:3] = terrain.center_position_stone[i,  0:3] + [i * self.env_length, j * self.env_width, 0]
            if terrain.idx == 17 and self.num_goals != None:
                self.goals_narrow[i, 0:3] = terrain.center_position[i, 0:3] + [(i+1) * self.env_length , j * self.env_width + self.env_width/2, 0.32]


        else:
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
