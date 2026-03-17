import os
import time
import copy
from collections import deque
import statistics

from termcolor import cprint

from torch.utils.tensorboard import SummaryWriter

import torch
from rl.env import VecEnv
from rl.MGDP.algorithms import PPO
from rl.MGDP.modules import ActorCritic, ActorCriticRecurrentLSTM

from rl.MGDP.modules.actor_critic_recurrent_lstm import ActorWrapper

class MGDPPolicyRunner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.encoder_cfg = train_cfg["Encoder"]

        self.encoder_mlp = train_cfg["Encoder"]['encoder_mlp_units']
        self.device = device
        self.env = env
        self.HistoryLen = train_cfg["Encoder"]['HistoryLen']

        ### add image shape ###
        self.MLPModule_info = train_cfg["Encoder"]['MLPModule_info']
        self.CNNModule_info = train_cfg["Encoder"]['CNNModule_info']
        # self.TranformModule_info = train_cfg["Encoder"]['TranformModule_info']
        # self.GRUModule_info = train_cfg["Encoder"]['GRUModule_info']
        # self.VAEModule_info = train_cfg["Encoder"]['VAEModule_info']
        # self.DecoderModule_info = train_cfg["Encoder"]['DecoderModule_info']


        self.camera_dim = train_cfg["Encoder"]['CNNModule_info']['output_channels']

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic

        if self.cfg["policy_class_name"] == "ActorCritic":
            actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                           self.env.num_actions,
                                                           **self.policy_cfg,
                                                           **self.encoder_cfg).to(self.device)

        elif self.cfg["policy_class_name"] == "ActorCriticRecurrentLSTM":
            ### add LSTM shape ###
            actor_critic: ActorCriticRecurrentLSTM = actor_critic_class(self.env.num_obs,
                                                           self.env.num_actions,
                                                           **self.policy_cfg,
                                                           **self.encoder_cfg).to(self.device)



        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic,
                                  device=self.device,
                                  **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_encoder_input = self.env.num_obs * self.HistoryLen
        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions],
                              [self.num_encoder_input], self.camera_dim)

        # Log
        self.log_dir = log_dir
        if log_dir is not None:
            self.nn_dir = os.path.join(self.log_dir, 'stage1_nn')
            self.tb_dir = os.path.join(self.log_dir, 'stage1_tb')
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=10)
        else:
            self.nn_dir = None
            self.tb_dir = None
            self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration_init = 0

        self.best_wm_visual_mse_loss = float('inf') 
        self.best_wm_combined_loss = float('inf')    
        self.best_avg_reward = -float('inf') 

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        self.alg.actor_critic.train() 

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.current_learning_iteration = self.current_learning_iteration_init
        start_iter = self.current_learning_iteration_init
        tot_iter = start_iter + num_learning_iterations



        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            current_iter_wm_visual_mse = 0.0
            current_iter_wm_height_mse = 0.0
            current_iter_wm_height_visual_mse = 0.0
            current_iter_wm_contrastive = 0.0
            current_iter_wm_visual_l1 = 0.0
            current_iter_wm_height_l1 = 0.0
            current_iter_total = 0.0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, next_obs=obs_dict['obs'], infos= infos)

                    if getattr(self.env, 'use_world_model', False):
                        current_iter_wm_visual_mse += self.env.visual_mse_loss.item()
                        current_iter_wm_height_mse += self.env.height_mse_loss.item()
                        current_iter_wm_height_visual_mse += self.env.height_visual_mse_loss.item()
                        current_iter_wm_contrastive += self.env.contrastive_loss.item()
                        current_iter_wm_visual_l1 += self.env.visual_l1_loss.item()
                        current_iter_wm_height_l1 += self.env.height_l1_loss.item()
                        current_iter_total += self.env.total_wm_loss.item()

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])


                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict)

            mean_value_loss, mean_surrogate_loss, mean_vel_loss, mean_feet_loss, mean_cnn_loss = self.alg.update()


 
            wm_visual_mse_loss = current_iter_wm_visual_mse / self.num_steps_per_env
            wm_height_mse_loss = current_iter_wm_height_mse / self.num_steps_per_env
            wm_height_visual_mse_loss = current_iter_wm_height_visual_mse / self.num_steps_per_env
            wm_contrastive_loss_loss = current_iter_wm_contrastive / self.num_steps_per_env
            wm_visual_l1_loss = current_iter_wm_visual_l1 / self.num_steps_per_env
            wm_height_l1_loss = current_iter_wm_height_l1 / self.num_steps_per_env
            wm_loss= current_iter_total  / self.num_steps_per_env

  
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.nn_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                if self.nn_dir is not None:
                    self.save(os.path.join(self.nn_dir, 'model_{}.pt'.format(it)))
                    self.save(os.path.join(self.nn_dir, 'last.pt'))

           
                    if len(rewbuffer) > 0: 
                        current_avg_reward = statistics.mean(rewbuffer)  
                        if current_avg_reward > self.best_avg_reward:
                            self.best_avg_reward = current_avg_reward 
                            self.save(os.path.join(self.nn_dir, 'model_best.pt')) 

                    self.save_world_model(os.path.join(self.nn_dir, 'wm_{}.pt'.format(it)))
                    self.save_world_model(os.path.join(self.nn_dir, 'wm_last.pt'))

                 
                    if getattr(self.env, 'use_world_model', False):
                        w_vis = getattr(self.env.cfg.camera, 'wm_best_visual_weight', 0.5)
                        w_h = getattr(self.env.cfg.camera, 'wm_best_height_weight', 0.5)
                        current_combined = w_vis * wm_visual_mse_loss + w_h * wm_height_mse_loss
                        if current_combined < self.best_wm_combined_loss:
                            self.best_wm_combined_loss = current_combined
                            self.best_wm_visual_mse_loss = wm_visual_mse_loss 
                            self.save_world_model(os.path.join(self.nn_dir, 'wm_best.pt'))
            ep_infos.clear()

        if self.nn_dir is not None:
            self.save(os.path.join(self.nn_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if self.writer is not None:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if self.writer is not None:
            self.writer.add_scalar('Actor/feet_loss', locs['mean_feet_loss'], locs['it'])
            self.writer.add_scalar('Actor/vel_loss', locs['mean_vel_loss'], locs['it'])
            self.writer.add_scalar('Actor/cnn_loss', locs['mean_cnn_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_loss', locs['wm_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_visual_mse_loss', locs['wm_visual_mse_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_height_mse_loss', locs['wm_height_mse_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_height_visual_mse_loss', locs['wm_height_visual_mse_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_contrastive_loss_loss', locs['wm_contrastive_loss_loss'], locs['it'])
            self.writer.add_scalar('Actor/wm_visual_l1_loss', locs['wm_visual_l1_loss'], locs['it'])
            self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
            self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            if self.writer is not None:
                self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration_init + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'actor_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def save_world_model(self, path, infos=None):
        depth_stats = {}
        if hasattr(self.env, "dynamic_min") and hasattr(self.env, "dynamic_max"):
            depth_stats = {
                "dynamic_min": float(self.env.dynamic_min),
                "dynamic_max": float(self.env.dynamic_max),
                "range_size": float(self.env.range_size),
                "depth_mean": float(getattr(self.env, "depth_mean", 0.0)),
                "depth_std": float(getattr(self.env, "depth_std", 1.0)),
            }

        state = {
            'image_encoder': self.env.cnn_encoder.state_dict(),
            'image_decoder': self.env.cnn_decoder.state_dict(),
            'wm_optimizer': self.env.wm_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
        }
        if depth_stats:
            state['depth_stats'] = depth_stats
        if getattr(self.env, 'use_map_decoder', False) and hasattr(self.env, 'map_encoder'):
            state['map_encoder'] = self.env.map_encoder.state_dict()
            state['map_decoder'] = self.env.map_decoder.state_dict()
        if getattr(self.env, 'use_memory', False) and hasattr(self.env, 'memory_lstm'):
            state['memory_encoder'] = self.env.memory_lstm.state_dict()
        torch.save(state, path)


    def export_to_onnx1(self, path, name):
        """将模型导出为 ONNX 格式"""
        os.makedirs(path, exist_ok=True)
        onnx_file_path = os.path.join(path, name)



        model_copy = copy.deepcopy(self.alg.actor_critic).to(self.device)
        actor_wrapper = ActorWrapper(model_copy)
        actor_wrapper.eval()



        batch_size = self.env.num_envs
        obs = torch.randn(batch_size, self.env.num_obs, device=self.device)
        camera_dim = self.encoder_cfg.get('camera_dim', self.encoder_cfg['CNNModule_info'].get('camera_dim', [16, 16]))
        if isinstance(camera_dim, int):
            cam_h = cam_w = camera_dim
        else:
            cam_h, cam_w = camera_dim[0], camera_dim[1]
        image_buf = torch.randn(batch_size, 2, cam_h, cam_w, device=self.device)
        proprio_hist = torch.randn(batch_size, self.num_encoder_input, device=self.device)
        feature_dim = self.encoder_cfg.get('feature_dim', self.encoder_cfg.get('GRUModule_info', {}).get('rnn_hidden_dims', 64))
        hidden_states = torch.randn(1, batch_size, feature_dim, device=self.device)



        input_names = ["observation", "image_buffer", "proprioceptive_history", "hidden_states"]
        output_names = ["action", "hidden_states_out"]

        torch.onnx.export(
            actor_wrapper,
            (obs, image_buf, proprio_hist, hidden_states),
            onnx_file_path,
            export_params=True,
            # do_constant_folding=True,
            # verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=16, 
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'image_buffer': {0: 'batch_size'},
                'proprioceptive_history': {0: 'batch_size'},
                'hidden_states': {1: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)

        self.alg.actor_critic.load_state_dict(loaded_dict['actor_state_dict'])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration_init = loaded_dict['iter']

        if self.cfg['export_policy'] == "onnx":
            import onnx
            cprint('Exporting policy to onnx module(C++)', 'red', attrs=['bold'])
            onnx_save_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'onnx_s1')
            os.makedirs(onnx_save_path, exist_ok=True) 

            from rl.utils.utils import (export_policy_as_onnx, export_cnn_as_onnx,
                                        export_cnn_decoder_as_onnx, export_map_decoder_as_onnx)
            export_policy_as_onnx(self.alg.actor_critic.actor,
                                  self.alg.actor_critic.num_actor_input,
                                  onnx_save_path,
                                  'actor.onnx',
                                  input_names=["observation"],
                                  output_names=["action"],
                                  )
            export_policy_as_onnx(self.alg.actor_critic.mlp_encoder,
                                  self.alg.actor_critic.mlp_input,
                                  onnx_save_path,
                                  'encoder.onnx',
                                  input_names=["obs_history"],
                                  output_names=["est_feature"],
                                  )

            cnn_network = self.env.cnn_encoder 
            input_shape = (1, 2, 16, 16)
            export_cnn_as_onnx(cnn_network,
                                  input_shape,
                                  onnx_save_path,
                                  'cnn.onnx',
                                   input_names=["imgae_shape"],
                                   output_names=["imgae_feature"],
                               )

     

            batch_size = 1
            input_channels = self.env.image_buf.shape[1]  
            img_height = self.env.image_buf.shape[2] 
            img_width = self.env.image_buf.shape[3]  
            cnn_input_size = (batch_size, input_channels, img_height, img_width)  
            save_path = onnx_save_path  
            cnn_file_name = 'cnn_encoder.onnx' 
            cnn_input_names = ["image_input"]  
            cnn_output_names = ["visual_token"] 
            export_cnn_as_onnx(
                cnn_network,
                cnn_input_size,
                save_path,
                cnn_file_name,
                cnn_input_names,
                cnn_output_names
            )

            self.env.cnn_encoder = self.env.cnn_encoder.to(self.device)
            cprint(f"Successfully exported cnn_encoder to {os.path.join(save_path, cnn_file_name)}", 'green')

            decoder_network = self.env.cnn_decoder  
            decoder_input_size = (1, 64)  
            decoder_save_path = onnx_save_path  
            decoder_file_name = 'cnn_decoder.onnx'  
            decoder_input_names = ["visual_token"] 
            decoder_output_names = ["recon_image"] 

            export_cnn_decoder_as_onnx(
                decoder_network,
                decoder_input_size,
                decoder_save_path,
                decoder_file_name,
                decoder_input_names,
                decoder_output_names
            )

            self.env.cnn_decoder = self.env.cnn_decoder.to(self.device)
            cprint(f"Successfully exported cnn_decoder to {os.path.join(decoder_save_path, decoder_file_name)}",
                   'green')

            if getattr(self.env, 'use_map_decoder', False) and hasattr(self.env, 'map_encoder'):
                map_encoder_network = self.env.map_encoder 
                batch_size = 1
                map_input_channels = 1 
                map_input_height = 17 
                map_input_width = 11 
                map_encoder_input_size = (batch_size, map_input_channels, map_input_height, map_input_width)
                map_encoder_file_name = 'map_encoder.onnx' 
                map_encoder_input_names = ["height_map_input"] 
                map_encoder_output_names = ["map_token"] 

                export_cnn_as_onnx(
                    map_encoder_network,
                    map_encoder_input_size,
                    onnx_save_path,
                    map_encoder_file_name,
                    map_encoder_input_names,
                    map_encoder_output_names
                )

                self.env.map_encoder = self.env.map_encoder.to(self.device)
                cprint(f"Successfully exported map_encoder to {os.path.join(onnx_save_path, map_encoder_file_name)}", 'green')

                map_decoder_network = self.env.map_decoder  
                map_decoder_input_size = (1, 32)  
                map_decoder_save_path = onnx_save_path  
                map_decoder_file_name = 'map_decoder.onnx' 
                map_decoder_input_names = ["map_token"] 
                map_decoder_output_names = ["recon_height_map"] 

                export_map_decoder_as_onnx(
                    map_decoder_network,
                    map_decoder_input_size,
                    map_decoder_save_path,
                    map_decoder_file_name,
                    map_decoder_input_names,
                    map_decoder_output_names
                )

                self.env.map_decoder = self.env.map_decoder.to(self.device)
                cprint(f"Successfully exported map_decoder to {os.path.join(map_decoder_save_path, map_decoder_file_name)}",
                       'green')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() 
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.evaluate

