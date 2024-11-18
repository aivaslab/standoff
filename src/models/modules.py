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

class NormalizedPerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 13)  # 2*6 + 1 = 13 outputs per timestep
        self.fc_presence = nn.Linear(5 * 5 * 7 * 7, 1) 
        
    def forward(self, x):
        treats_list = []
        vision_list = []
        
        batch_size = x.shape[0]
        
        for timestep in range(5):
            current = x[:, timestep]  # [batch, channel, height, width]
            h = F.relu(self.conv1(current))
            h = F.relu(self.conv2(h))
            h = h.flatten(1)
            h = self.fc(h)
            
            treats = h[:, :12].view(batch_size, 2, 6)  # [batch, 2, 6]
            vision = h[:, 12]  # [batch]
            
            treats = F.softmax(treats, dim=-1)
            vision = torch.sigmoid(vision)
            
            treats_list.append(treats)
            vision_list.append(vision)
        
        treats = torch.stack(treats_list, dim=1)  # [batch, timestep, 2, 6]
        vision = torch.stack(vision_list, dim=1)  # [batch, timestep]
        
        presence = torch.sigmoid(self.fc_presence(x.reshape(batch_size, -1)))
        
        return {
            'treats_visible': treats,
            'opponent_vision': vision,
            'opponent_presence': presence
        }

class PerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True):
        super().__init__(use_neural)
        
    def _create_neural_network(self) -> nn.Module:
        return NormalizedPerceptionNetwork()
    
    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = perceptual_field.shape[0]

        for timestep in range(5):
            for layer in [2, 3]:
                indices = torch.nonzero(perceptual_field[0, 4, 2:4])

                for pos in indices:
                    print(f"timestep {pos[0]}, layer {pos[1] + 2}, value:", perceptual_field[0, 4, pos[1] + 2, pos[0], pos[2]])

        treats_visible = torch.randn(batch_size, 5, 2, 6)  # [batch, timestep, large/small, position]
        treats_visible = F.softmax(treats_visible, dim=-1) # this was from random initialization here

        # channel 0 is agents
        # channel 1 is boxes... always 3?
        # channel 2 is treat1
        # channel 3 is treat2
        # channel 4 is walls

        y_index = 5 #idk what this should be
        x_index = 5


        opponent_vision = torch.randn(batch_size, 5)       # [batch, timestep]
        opponent_presence = torch.randn(batch_size, 1) 
        
        # Todo: Fill in treat finding
        return {
            'treats_visible': treats_visible,
            'opponent_vision': opponent_vision,
            'opponent_presence': opponent_presence
        }

# Belief module
# Inputs: Visible Treats, Vision (1s if subject)
# Outputs: Belief vector at last timestep (2x6, normalized)
# Method: Start at the last timestep, and step backwards until we find the last visible instance of each treat 

class NormalizedBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(65, 32)
        self.fc2 = nn.Linear(32, 12)
        
    def forward(self, treats, vision):
        batch_size = treats.shape[0]
        x = torch.cat([treats.reshape(batch_size, -1), vision], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, 2, 6)
        x = F.softmax(x, dim=1)
        return x

class BeliefModule(BaseModule):
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
                    if vision[batch, t] > 0.5 and visible_treats[batch, t, treat_type, :5].max() > 0.5:
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

class DecisionModule(BaseModule):
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
                large_treats[dom_idx] = float('-inf')
                small_treats[dom_idx] = float('-inf')
                
                large_best = torch.argmax(large_treats[:5])
                if large_best != 5:
                    decisions.append(F.one_hot(large_best, num_classes=5))
                else:
                    small_best = torch.argmax(small_treats[:5])
                    decisions.append(F.one_hot(small_best, num_classes=5))
        
        return torch.stack(decisions).float().to(device)

# Final output module
# Inputs: Opponent presence, Greedy decision (sub), Sub decision
# Outputs: The correct label
# Greedy decision if no opponent, sub decision if opponent

class FinalOutputModule(BaseModule):
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
            batch_size = opponent_presence.shape[0]
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
        return sub_decision if opponent_presence else greedy_decision


class AblationArchitecture(nn.Module):
    def __init__(self, module_configs: Dict[str, bool]):
        super().__init__()
        self.kwargs = {'module_configs': module_configs}
        self.perception = PerceptionModule(use_neural=module_configs.get('perception', True))
        self.my_belief = BeliefModule(use_neural=module_configs.get('my_belief', True))
        self.op_belief = BeliefModule(use_neural=module_configs.get('op_belief', True))
        self.my_greedy_decision = DecisionModule(is_subordinate=False, use_neural=module_configs.get('my_greedy_decision', True))
        self.op_greedy_decision = DecisionModule(is_subordinate=False, use_neural=module_configs.get('op_greedy_decision', True))
        self.sub_decision = DecisionModule(is_subordinate=True, use_neural=module_configs.get('sub_decision', True))
        self.final_output = FinalOutputModule(use_neural=module_configs.get('final_output', True))
    
    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor) -> torch.Tensor:

        #print('Perception input', perceptual_field[0])

        perception_output = self.perception.forward(perceptual_field)
        device = perceptual_field.device
        #print('Perception output', perception_output[0])
        treats_visible = perception_output['treats_visible']
        opponent_vision = perception_output['opponent_vision']
        opponent_presence = perception_output['opponent_presence']
        #print("Opponent presence distribution:", opponent_presence.mean().item())

        batch_size = perceptual_field.shape[0]
        my_belief_vector = self.my_belief.forward(treats_visible, torch.ones(batch_size, 5, device=device))
        #print("My belief argmax distribution:", torch.argmax(my_belief_vector, dim=-1).unique(return_counts=True))
        op_belief_vector = self.op_belief.forward(treats_visible, opponent_vision)
        
        my_greedy_decision = self.my_greedy_decision.forward(my_belief_vector)
        #print("My greedy decision distribution:", torch.bincount(my_greedy_decision))
        op_greedy_decision = self.op_greedy_decision.forward(op_belief_vector)
        #print("Op greedy decision distribution:", torch.bincount(op_greedy_decision))

        sub_decision = self.sub_decision.forward(my_belief_vector, op_greedy_decision)
        #print("Sub decision distribution:", torch.bincount(sub_decision))
        
        final_decision = self.final_output.forward(opponent_presence, my_greedy_decision, sub_decision)
        #print("Final decision distribution:", torch.bincount(final_decision))
        
        return final_decision


if __name__ == "__main__":
    use_neural = False
    configs = {
        'perception': use_neural,
        'my_belief': use_neural,
        'op_belief': use_neural,
        'my_greedy_decision': use_neural,
        'op_greedy_decision': use_neural,
        'sub_decision': use_neural,
        'final_output': use_neural
    }

    network = AblationArchitecture(configs)
    dummy_input = torch.randn(50, 5, 5, 7, 7)
    output = network.forward(dummy_input, dummy_input)
    print(output)