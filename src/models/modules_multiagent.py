import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from typing import Dict

from .modules import *


class VisionPerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_detector = nn.Sequential(
            nn.Linear(5, 5),
        )

    def forward(self, x):
        batch_size, timesteps, grid_h, grid_w = x.shape[:4]
        
        x = x.reshape(batch_size * grid_h * grid_w, timesteps)
        x = torch.sigmoid_(self.vision_detector(x))
        return x.view(batch_size, grid_h, grid_w, timesteps)

class MultiVisionPerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)


    def _create_neural_network(self) -> nn.Module:
        return VisionPerceptionNetwork()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = perceptual_field.shape
        device = perceptual_field.device
    
        offset = torch.ones(width, device=device) * 2
        offset[4:] = -2
        
        dest_x = (torch.arange(width, device=device) + offset).long()
        vision_channel = perceptual_field[:, :, 4]
        reindexed = vision_channel[:, :, :, dest_x]
        
        vision = torch.sigmoid_(20 * (torch.abs(reindexed - 1.0) - 0.5))

        return vision

    def _random_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = perceptual_field.shape
        device = perceptual_field.device
        
        return torch.randint(0, 2, (batch_size, timesteps, height, width), device=device).float()


class PresencePerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.presence_detector = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, x):
        batch_size, grid_h, grid_w = x.shape[:3]
        
        x = x.reshape(batch_size * grid_h * grid_w, 1)
        x = torch.sigmoid_(5*self.presence_detector(x))
        return x.view(batch_size, grid_h, grid_w)

class MultiPresencePerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return PresencePerceptionNetwork()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels, height, width = perceptual_field.shape
        
        presence = perceptual_field[:, 0, 0]
        
        return presence

    def _random_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width = perceptual_field.shape
        device = perceptual_field.device
        
        return torch.randint(0, 2, (batch_size, height, width), device=device).float()

class MultiAgentArchitecture(nn.Module):
    def __init__(self, module_configs: Dict[str, bool], random_probs: Dict[str, float] = None, grid_size=7, batch_size=256):
        super().__init__()
        print("init MultiAgentArchitecture")
        self.kwargs = {'module_configs': module_configs}
        self.kwargs['batch_size'] = batch_size
        self.grid_size = grid_size
        self.vision_prob_base = module_configs.get('vision_prob', 1.0)
        self.vision_prob = self.vision_prob_base
        self.num_visions = module_configs.get('num_beliefs', 1)
        self.detach_treat = module_configs['shared_treat'] and module_configs['my_treat'] and module_configs['detach']
        self.detach_belief = module_configs['shared_belief'] and module_configs['my_belief'] and module_configs['detach']
        self.detach_decision = module_configs['shared_decision'] and module_configs['my_decision'] and module_configs['detach']
        self.detach_combiner = module_configs['shared_combiner'] and module_configs['combiner'] and module_configs['detach']

        sigmoid_temp = module_configs.get('sigmoid_temp', 50.0)

        self.register_buffer('null_decision', torch.zeros(self.kwargs['batch_size'], 5))
        self.register_buffer('null_presence', torch.zeros(self.kwargs['batch_size'], 1))

        print(module_configs['vision_my'])

        if random_probs is None:
            random_probs = {k: 0.0 for k in module_configs.keys()}

        self.treat_perception = TreatPerceptionModule(
            use_neural=module_configs['my_treat'],
            random_prob=random_probs['my_treat'],
            sigmoid_temp=sigmoid_temp
        )

        self.vision_perception = MultiVisionPerceptionModule(
            use_neural=module_configs['vision_my'],
            random_prob=random_probs['vision_my'],
            sigmoid_temp=sigmoid_temp
        )

        self.presence_perception = MultiPresencePerceptionModule(
            use_neural=module_configs['presence_my'],
            random_prob=random_probs['presence_my'],
            sigmoid_temp=sigmoid_temp
        )

        self.belief = BeliefModule(
            use_neural=module_configs['my_belief'],
            random_prob=random_probs['my_belief'],
            sigmoid_temp=sigmoid_temp, 
            uncertainty=0.3
        )

        self.decision = DecisionModule(
            use_neural=module_configs['my_decision'],
            random_prob=random_probs['my_decision'],
            sigmoid_temp=sigmoid_temp
        )

    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device


        treats = self.treat_perception(perceptual_field)
        vision = self.vision_perception(perceptual_field)
        presence = self.presence_perception(perceptual_field)

        B, T, num_treats, treat_pos = treats.shape
        H, W = perceptual_field.shape[3], perceptual_field.shape[4]

        treats_expanded = treats.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, H, W, 1)
        treats_flat = treats_expanded.permute(0, 3, 4, 1, 2, 5).reshape(B*H*W, T, num_treats, treat_pos)
        vision_flat = vision.permute(0, 2, 3, 1).reshape(B*H*W, T)
        self.belief.uncertainty = 0.0

        belief_l_flat = self.belief(treats_flat[:, :, 0], vision_flat)
        belief_s_flat = self.belief(treats_flat[:, :, 1], vision_flat)

        beliefs_flat = torch.stack([belief_l_flat, belief_s_flat], dim=1)
        grid_beliefs = beliefs_flat.reshape(B, H, W, 2, 6)

        null_decision = torch.zeros(B, 5, device=device).unsqueeze(1).unsqueeze(1).expand(B, H, W, 5)
        null_decision_flat = null_decision.reshape(B*H*W, 5)
        
        null_presence = torch.zeros(B, 1, device=device).unsqueeze(1).unsqueeze(1).expand(B, H, W, 1)
        null_presence_flat = null_presence.reshape(B*H*W, 1)


        combined_beliefs_flat = grid_beliefs.reshape(B*H*W, 2, 6)
        dominant_decisions_flat = self.decision(combined_beliefs_flat, null_decision_flat, null_presence_flat)
        dominant_decisions = dominant_decisions_flat.reshape(B, H, W, 5) 

        dominant_flat = dominant_decisions.view(B, H*W, 5)
        presence_flat = presence.view(B, H*W)

        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, H*W)

        masked_presence = presence_flat.clone()
        presence_matrix = presence_flat.unsqueeze(2) * presence_flat.unsqueeze(1)

        identity = torch.eye(H*W, device=device).unsqueeze(0)
        non_identity = 1.0 - identity

        not_self_but_other = presence_matrix * non_identity
        decisions_expanded = dominant_flat.unsqueeze(1).expand(-1, H*W, -1, -1)
        masked_expanded = decisions_expanded * not_self_but_other.unsqueeze(-1)
        max_decisions = masked_expanded.max(dim=2)[0]

        #max_decisions = max_decisions * presence_flat.unsqueeze(-1)
        max_decisions_flat = max_decisions.reshape(B*H*W, 5)

        
        #dom_exists = torch.sigmoid_(20 * (-0.5 + max_decisions_flat.sum(dim=-1).reshape((B*H*W, 1))))
        dom_exists = (max_decisions_flat.sum(dim=-1) > 0).float().unsqueeze(1)

        print(dom_exists.shape, max_decisions_flat.shape, combined_beliefs_flat.shape)
        print(dom_exists.reshape(B, H, W, 1)[0])

        subordinate_decisions_flat = self.decision(combined_beliefs_flat, max_decisions_flat, dom_exists)
        subordinate_decisions = subordinate_decisions_flat.reshape(B, H, W, 5)
        

        op_grid = max_decisions.view(B, H, W, 5)

        print("Grid beliefs at different positions:")
        for i in range(7):
            print(f"Position (3,{i}):", grid_beliefs[0, 3, i])

                
        return {
            'treat_perception': treats,
            'vision_perception': vision[:, :, 3, 0],
            'vision_perception_my': vision[:, :, 3, 6],
            'presence_perception': presence[:, 3, 0].unsqueeze(-1),
            'my_decision': subordinate_decisions[:, 3, 6],
            'op_decision': op_grid[:, 3, 0],
            'my_belief': grid_beliefs[:, 3, 6],
            'op_belief': grid_beliefs[:, 3, 0],
        }

        # where could it fail?
        # final subordinate decision
        # dominant decision (let's check!)