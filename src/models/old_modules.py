import torch
import torch.nn as nn
from abc import ABC 
import torch.nn.functional as F
from typing import Dict


class BaseModule(nn.Module, ABC):
    def __init__(self, use_neural: bool = True):
        super().__init__()
        self.use_neural = use_neural
        self.neural_network = self._create_neural_network() if use_neural else None
    
    def _create_neural_network(self) -> nn.Module:
        pass
    
    def _hardcoded_forward(self, *args, **kwargs):
        pass
    
    def forward(self, *args, **kwargs):
        if self.use_neural:
            return self.neural_network(*args, **kwargs)
        return self._hardcoded_forward(*args, **kwargs)

# Perception module
# Inputs: The original perceptual field (5x5x7x7)
# Outputs: Visible treats (2x6 length vectors at each of 5 timesteps, normalized (position 5 is nothing)), opponent vision (scalar at each timestep), opponent presence (scalar overall) [56 total bits]
# Method: Find where treats are 1, where vision is shown, where opponent is.


class PerceptionModuleOld(BaseModule):
    def __init__(self, use_neural: bool = True):
        super().__init__(use_neural)
        
    def _create_neural_network(self) -> nn.Module:
        return NormalizedPerceptionNetwork()
    
    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        # channel 0 is agents
        # channel 1 is boxes... always 3?
        # channel 2 is treat1, at x=3 and y=1-6
        # channel 3 is treat2, at x=3 and y=1-6
        # channel 4 is walls

        #batch_size = perceptual_field.shape[0]

        #device = perceptual_field.device

        treats_visible = perceptual_field[:, :, 2:4, 1:6, 3] > 0
        treats_visible = torch.flip(treats_visible, dims=[3]) # AAAAAHHH
        opponent_vision = perceptual_field[:, :, 4, 3, 2] != 1
        #print(perceptual_field[:, :, 4, 3, 2])
        opponent_presence = perceptual_field[:, 0, 0, 3, 0].unsqueeze(1)

        # this adds the 6th treat position when the others are empty
        no_treats = ~treats_visible.any(dim=3, keepdim=True)
        treats_visible = torch.cat([treats_visible, no_treats], dim=3)

        #print('sizes', treats_visible.shape, opponent_vision.shape, opponent_presence.shape)

        #print('field', perceptual_field[0, 0, 0])

        #print('perception:', treats_visible[0], opponent_vision[0], opponent_presence[0])

        return {
            'treats_visible': treats_visible,
            'opponent_vision': opponent_vision,
            'opponent_presence': opponent_presence
        }

# Belief module
# Inputs: Visible Treats, Vision (1s if subject)
# Outputs: Belief vector at last timestep (2x6, normalized)
# Method: Start at the last timestep, and step backwards until we find the last visible instance of each treat 



class BeliefModuleOld(BaseModule):
    def __init__(self, use_neural: bool = True):
        super().__init__(use_neural)
    
    def _create_neural_network(self) -> nn.Module:
        return NormalizedBeliefNetwork()
    
    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        batch_size = visible_treats.shape[0]
        device = visible_treats.device
        belief_vectors = []

        for batch in range(batch_size):
            belief_vector = torch.zeros(2, 6, device=device)
            
            for treat_type in range(2):
                found = False
                for t in range(4, -1, -1):
                    if vision[batch, t] and visible_treats[batch, t, treat_type, :5].max() > 0.5:
                        belief_vector[treat_type] = visible_treats[batch, t, treat_type]
                        found = True
                        break
                
                if not found:
                    belief_vector[treat_type, 5] = 1
                    
            belief_vectors.append(belief_vector)

        return torch.stack(belief_vectors)

# Greedy Decision module
# Inputs: Belief vector
# Output: Decision
# Method: Argmax of the large treat belief unless it is 5, else argmax of small treat belief unless it is 5, else 2

class DecisionModuleOld(BaseModule):
    def __init__(self, is_subordinate: bool = False, use_neural: bool = True):
        self.is_subordinate = is_subordinate
        super().__init__(use_neural)
    
    def _create_neural_network(self) -> nn.Module:
        input_size = 12 if not self.is_subordinate else 12 + 5
        return nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Softmax(dim=-1)
        )

    def forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None) -> torch.Tensor:
        batch_size = belief_vector.shape[0]
        if self.use_neural:
            if self.is_subordinate:
                x = torch.cat([belief_vector.view(batch_size, -1), dominant_decision], dim=-1)
            else:
                x = belief_vector.reshape(batch_size, -1)
            return self.neural_network(x)
        return self._hardcoded_forward(belief_vector, dominant_decision)
    
    def _hardcoded_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None) -> torch.Tensor:
        batch_size = belief_vector.shape[0]
        device = belief_vector.device
        decisions = []
        
        for batch in range(batch_size):
            if not self.is_subordinate:
                large_treats = belief_vector[batch, 0]
                small_treats = belief_vector[batch, 1]

                if torch.argmax(large_treats) != 5:
                    decisions.append(F.one_hot(torch.argmax(large_treats), num_classes=5))
                elif torch.argmax(small_treats) != 5:
                    decisions.append(F.one_hot(torch.argmax(small_treats), num_classes=5))
                else:
                    decisions.append(F.one_hot(torch.tensor(2, device=device), num_classes=5))
            else:
                large_treats = belief_vector[batch, 0].clone()
                small_treats = belief_vector[batch, 1].clone()
                dom_idx = torch.argmax(dominant_decision[batch]).long()
                
                large_best = torch.argmax(large_treats[:5])
                if large_best != dom_idx: # if the dominant isn't going large, we do
                    decisions.append(F.one_hot(large_best, num_classes=5))
                else: # otherwise, dominant gets large, so we get small
                    small_best = torch.argmax(small_treats[:5])
                    decisions.append(F.one_hot(small_best, num_classes=5))
        #print('decision', self.is_subordinate, decisions[0])
        return torch.stack(decisions).float().to(device)

# Final output module
# Inputs: Opponent presence, Greedy decision (sub), Sub decision
# Outputs: The correct label
# Greedy decision if no opponent, sub decision if opponent

class FinalOutputModuleOld(BaseModule):
    def __init__(self, use_neural: bool = True):
        super().__init__(use_neural)
    
    def _create_neural_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU(),
            nn.Linear(16, 5), 
            nn.Softmax(dim=-1)
        )

    def forward(self, opponent_presence: torch.Tensor, greedy_decision: torch.Tensor, sub_decision: torch.Tensor) -> torch.Tensor:
        if self.use_neural:
            #batch_size = opponent_presence.shape[0]
            x = torch.cat([opponent_presence, greedy_decision, sub_decision], dim=1)
            probs = self.neural_network(x)
            return probs
        else:
            device = opponent_presence.device
            decisions = []
            for op, gd, sd in zip(opponent_presence, greedy_decision, sub_decision):
                decisions.append(sd if op > 0.5 else gd)
            return torch.stack(decisions).to(device)
    
    def _hardcoded_forward(self, opponent_presence: torch.Tensor, greedy_decision: torch.Tensor, sub_decision: torch.Tensor) -> torch.Tensor:
        return torch.where(
            opponent_presence.unsqueeze(1),
            sub_decision,
            greedy_decision
        )
