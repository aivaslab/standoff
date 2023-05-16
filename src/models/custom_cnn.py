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

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 16, conv_mult: int = 1, frames: int = 1, label_dim=0):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = self.get_image_obs(observation_space).shape[0] - 1 if label_dim > 0 else self.get_image_obs(observation_space).shape[0]
        self.label_dim = label_dim

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
                th.as_tensor(self.get_image_obs(observation_space).sample()[None]).float()[:, :-1 if self.label_dim > 0 else None]
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + self.label_dim, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image_obs = self.get_image_obs(observations)
        #print(image_obs.shape)
        features = self.cnn(image_obs[:, :-1 if self.label_dim > 0 else None])
        #print(features.shape)
        if self.label_dim > 0:
            labels = image_obs[:, -1, 0, :self.label_dim]
            #print(labels.shape)

            features = th.cat([features, labels], dim=1)

        return self.linear(features)

    def get_image_obs(self, observations):
        return observations['image'] if type(observations) is dict else observations

