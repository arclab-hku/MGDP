from torch.distributions import Normal
import torch.nn as nn
import torch
from .Network import *
from termcolor import cprint
import numpy as np

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        output_nums = kwargs['MLPModule_info']['hidden_dims'][2]
        camera_dim = kwargs.get('CNNModule_info', {}).get('output_channels', 64)

        self.num_actor_input = num_obs + camera_dim + output_nums
        self.num_critic_input = num_obs + output_nums + 187  # 45 + 7 + 187 = 239（vel 3 + foot 4 + height 187）

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.mlp_input = kwargs["MLPModule_info"]["input_dims"]
        self.hidden_dims = kwargs["MLPModule_info"]["hidden_dims"]
        self.mlp_encoder = MLPModule(self.mlp_input,
                                     self.hidden_dims)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, e = self._actor_critic(obs_dict)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        _, _, value, extrin = self._actor_critic(obs_dict)
        return value, extrin


    def _actor_critic(self, obs_dict):

        obs_vel = obs_dict['privileged_info'][:, 0:3]
        obs_foot_height = obs_dict['privileged_info'][:, 3:7]
        obs_height = obs_dict['privileged_info'][:, 7:]

        proprioceptive_token = self.mlp_encoder(obs_dict['proprio_hist'])

        visual_token = obs_dict['image_buf']
        visual_tokens_height, visual_tokens_image = visual_token[:, 0:32], visual_token[:, 32:]

        actor_obs = torch.cat([obs_dict['obs'], proprioceptive_token[:, 0:3],  proprioceptive_token[:, 3:],
                               visual_tokens_height, visual_tokens_image],  dim=-1)  ## 45 + 3 + 4 + 16


        critic_obs = torch.cat([obs_dict['obs'], obs_vel, obs_foot_height, obs_height], dim=-1)  ## 45+7+16 + 4 + 31 = 235

        extrin = [proprioceptive_token, visual_tokens_height, visual_tokens_image]

        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std

        return mu, mu * 0 + sigma, value, extrin

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


