from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import gym
import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 16, conv_mult: int = 1, frames: int = 1, use_label=False):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = self.get_image_obs(observation_space).shape[0]
        self.n_label_channels = len(self.get_label_obs(observation_space)) if use_label else 0
        print('n_label_channels', self.n_label_channels)
        #print(self.get_image_obs(observation_space).shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4*conv_mult*frames, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(4*conv_mult*frames, 8*conv_mult, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(self.get_image_obs(observation_space).sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + self.n_label_channels, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image_features = self.cnn(self.get_image_obs(observations))

        if self.n_label_channels > 0:
            features = th.cat((image_features, observations['label']), dim=1)
        else:
            features = image_features

        return self.linear(features)

    def get_image_obs(self, observations):
        return observations['image'] if type(observations) is dict else observations

    def get_label_obs(self, observations):
        return observations['label'] if type(observations) is dict else []
