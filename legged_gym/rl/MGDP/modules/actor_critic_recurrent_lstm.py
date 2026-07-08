import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import ActorCritic, get_activation
from .Network import *
from termcolor import cprint
import numpy as np


class ActorCriticRecurrentLSTM(ActorCritic):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):

        super().__init__(num_obs,
                         num_actions,
                         actor_hidden_dims,
                         critic_hidden_dims,
                         activation,
                         init_noise_std,
                         **kwargs)

        self.pie_encoder = PIEEncoder(kwargs['MLPModule_info'],
                                      kwargs['CNNModule_info'],
                                      kwargs['TranformModule_info'],
                                      kwargs['GRUModule_info'],
                                      kwargs['VAEModule_info'],
                                      kwargs['ProjectionHead_info'],
                                      kwargs['DecoderModule_info']
                                      )

        cprint(f"pie_encoder: {self.pie_encoder}", 'red', attrs=['bold'])

        # self.memory_lstm = self.pie_encoder.gru
        # print(f"RNN: {self.memory_lstm}")
        self.height_encoder = NEWCNNHieghtEncoder(kwargs['cnn_mlp_units'][0])
        self.depth_precent = kwargs['depth_precent']

    # def reset(self, dones=None):
    #     self.memory_lstm.reset(dones)

    # def get_hidden_states(self):
    #     return self.memory_lstm.hidden_states

    def act(self, obs_dict, masks=None, hidden_states=None):
        mean, std, _, e = self._actor_critic(obs_dict, masks=masks, hidden_states=hidden_states)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions, masks=None, hidden_states=None):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, masks=None, hidden_states=None):
        # used for testing
        actions_mean, _, _, _ = self._actor_critic(obs_dict, masks=masks, hidden_states=hidden_states)
        return actions_mean

    def evaluate(self, obs_dict, masks=None, hidden_states=None):
        _, _, value, extrin = self._actor_critic(obs_dict, masks=masks, hidden_states=hidden_states)
        return value, extrin

    def _actor_critic(self, obs_dict, masks=None, hidden_states=None):
        obs_vel = obs_dict['privileged_info'][:, 0:3]
        obs_foot_height = obs_dict['privileged_info'][:, 3:7]
        obs_height = obs_dict['privileged_info'][:, 7:]

        proprioceptive_token, visual_token = self.pie_encoder(obs_dict['image_buf'], obs_dict['proprio_hist'])


        visual_tokens_height, visual_tokens_image = visual_token[:, 0:16], visual_token[:, 16:]


        if self.depth_precent[1]==1:
            extrin_height = None

        else:
            n = int((obs_height.shape[1] / 187) ** 0.5)
            batch_size = obs_height.size(0)
            obs_height_new = obs_height.view(batch_size, 1, n * 17, n * 11)  # 或者使用 torch.reshape
            extrin_height = self.height_encoder(obs_height_new)

        random_integer = np.random.choice([0, 1], p=self.depth_precent)  # 可能输出 0 或 1

        if random_integer == 1:
            actor_obs = torch.cat([obs_dict['obs'], proprioceptive_token, visual_tokens_height, visual_tokens_image],
                                  dim=-1)  ## 45 + 3 + 4 + 16
        else:
            actor_obs = torch.cat([obs_dict['obs'], proprioceptive_token, extrin_height, visual_tokens_image],
                                  dim=-1)  ## 45 + 3 + 4 + 16

        critic_obs = torch.cat([obs_dict['obs'], obs_vel, obs_foot_height, obs_height], dim=-1)  # 45+3+4+187=239，与父类 critic 输入一致

        extrin = [proprioceptive_token, visual_tokens_height, visual_tokens_image, extrin_height]

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std

        return mu, mu * 0 + sigma, value, extrin

class ActorWrapper(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.pie_encoder = actor_critic.pie_encoder
        self.memory_lstm = actor_critic.memory_lstm
        self.actor = actor_critic.actor

    def forward(self, obs, image_buf, proprio_hist, hidden_states):
        transform_out = self.pie_encoder(image_buf, proprio_hist)
        # print('extrin_obs_z', transform_out.shape)

        gru_output, hidden_states = self.memory_lstm.forward_onnx(transform_out, None, hidden_states)
        # print('gru_output', gru_output.shape)
        extrin_encoder, extrin_height, extrin_obs = gru_output[:, 0:7], gru_output[:, 7:7 + 16], gru_output[:, 7 + 16:]
        extrin_obs_z, mean, logvar = self.pie_encoder.vae(extrin_obs)
        # print('extrin_obs_z', extrin_encoder.shape, extrin_obs_z.shape, extrin_height.shape)
        actor_obs = torch.cat([obs, extrin_encoder, extrin_obs_z, extrin_height], dim=-1)
        # print('actor_obs', actor_obs.shape)
        mu = self.actor(actor_obs)
        return mu, hidden_states

    def act_inference(self, obs_dict, hidden_states):
        """用于测试的接口，接受obs_dict格式的输入"""
        obs = obs_dict['obs']
        image_buf = obs_dict['image_buf']
        proprio_hist = obs_dict['proprio_hist']

        actions_mean, hidden_states = self.forward(obs, image_buf, proprio_hist, hidden_states)
        return actions_mean, hidden_states


