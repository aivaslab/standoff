import torch
import torch.nn as nn
from abc import ABC 
import torch.nn.functional as F
from typing import Dict
from .old_modules import PerceptionModuleOld, BeliefModuleOld, FinalOutputModuleOld, DecisionModuleOld

def softargmax(x, dim=-1, temperature=1.0):
    return F.softmax(x / temperature, dim=dim).mul(torch.arange(x.size(dim)).to(x.device)).sum(dim=dim)

def differentiable_one_hot(x, num_classes=5, temperature=0.05):
    #return F.one_hot(x, num_classes = num_classes)
    scaled = -((torch.arange(num_classes, device=x.device).float().unsqueeze(0) - x.unsqueeze(1)) ** 2) / temperature
    return F.softmax(scaled, dim=1)

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
        # channel 0 is agents
        # channel 1 is boxes... always 3?
        # channel 2 is treat1, at x=3 and y=1-6
        # channel 3 is treat2, at x=3 and y=1-6
        # channel 4 is walls

        #batch_size = perceptual_field.shape[0]

        #device = perceptual_field.device

        #treats_visible = perceptual_field[:, :, 2:4, 1:6, 3] > 0

        treats_visible = torch.sigmoid(50 * (perceptual_field[:, :, 2:4, 1:6, 3] - 0.5)) # uses a sharp sigmoid to retain gradients instead of > 0

        treats_visible = torch.flip(treats_visible, dims=[3]) # AAAAAHHH
        #opponent_vision = perceptual_field[:, :, 4, 3, 2] != 1

        #opponent_vision = torch.sigmoid(50 * torch.abs(perceptual_field[:, :, 4, 3, 2] - 1.0) - 0.5) # likewise a sharp sigmod and abs for differentiable "!="
        opponent_vision = torch.sigmoid(50 * (torch.abs(perceptual_field[:, :, 4, 3, 2] - 1.0) - 0.5))
        #print(perceptual_field[:, :, 4, 3, 2])
        opponent_presence = perceptual_field[:, 0, 0, 3, 0].unsqueeze(1)

        # this adds the 6th treat position when the others are empty
        #no_treats = ~treats_visible.any(dim=3, keepdim=True)

        treat_sums = treats_visible.sum(dim=3, keepdim=True)
        no_treats = torch.sigmoid(50 * (0.1 - treat_sums))  # close to 1 when sum is close to 0, and differentiable

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

class NormalizedBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(65, 32)
        self.fc2 = nn.Linear(32, 12)
        
    def forward(self, treats, vision):
        batch_size = treats.shape[0]
        treats_flat = treats.reshape(batch_size, -1)
        vision_shaped = vision.reshape(batch_size, -1)

        x = torch.cat([treats_flat, vision_shaped], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, 2, 6)
        x = F.softmax(x, dim=-1)
        return x

class BeliefModule(BaseModule):
    def __init__(self, use_neural: bool = True):
        super().__init__(use_neural)
    
    def _create_neural_network(self) -> nn.Module:
        return NormalizedBeliefNetwork()

    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:

        batch_size = visible_treats.shape[0]
        device = visible_treats.device

        #time_weights = torch.exp(torch.arange(5, device=device) * 20.0).unsqueeze(-1)

        # so for treat 1, position beliefs are all the same, despite ever see treat being 1. time weights are in the range 1-2^30.
        # has treat was 1, 0, 0, 0, 0

        # so position beliefs are broken. They're a sum of weighted positions. Weighted positions was positions (assumed correct because has treat) and
        # Okay so the later position beliefs are far too large.

        # Next iteration. Op beliefs are selecting non 5 despite no vision. Why? Ever see treat is incorrectly 1. Or wrong index.
        # no, ever see treat with 0 vision is becoming 4.5 e-5. Is that too big? Time weights scale from -16 to -12. And then no treat belief becomes 0.2, should be 1. Made it 1.

        beliefs = []
        for treat_type in range(2):
            treats = visible_treats[:, :, treat_type]  # [batch, 5, 6]
            positions = treats[..., :5]  # [batch, 5, 5]

            # First determine if each timestep sees a treat
            has_treat = torch.sigmoid(20 * (positions.max(dim=-1)[0] - 0.5))  # [batch, 5]
            #print('has treat', treat_type, has_treat[0])
            #print('vision', vision[0])

            # Mask with vision
            valid_observations = has_treat * vision  # [batch, 5]
            # Calculate if we ever see a treat across all timesteps
            ever_see_treat = torch.sigmoid(20 * (valid_observations.max(dim=1)[0] - 0.5))  # [batch]
            #print('ever_see_treat', treat_type, ever_see_treat[0])

            # If we ever see a treat, create position beliefs using time weights
            time_weights = torch.exp(torch.arange(5, device=device) * 2.0).view(1, 5)  # [1, 5]
            time_weights = time_weights * valid_observations
            #print('time_weights', treat_type, time_weights[0])
            weighted_positions = positions * time_weights.unsqueeze(-1)  # [batch, 5, 5]
            position_beliefs = weighted_positions.sum(dim=1)  # [batch, 5]
            #print('position_beliefs', treat_type, position_beliefs[0])
            position_beliefs = position_beliefs / (time_weights.sum(dim=1, keepdim=True) + 1e-10)  # [batch, 5]
            #print('position_beliefs rescaled', treat_type, position_beliefs[0])

            no_treat_belief = (1 - ever_see_treat).unsqueeze(-1)  # [batch, 1]
            #print('no_treat_belief', treat_type, no_treat_belief[0])
            belief = torch.cat([position_beliefs, no_treat_belief], dim=1)  # [batch, 6]

            # Normalize with stability
            #belief = belief + 1e-10
            belief = belief / belief.sum(dim=1, keepdim=True)

            beliefs.append(belief)

        result = torch.stack(beliefs, dim=1)
        return result
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

        if not self.is_subordinate:
            default_choice = torch.zeros(batch_size, 5, device=device)
            default_choice[:, 2] = 1.0

            large_no_treat = torch.sigmoid(50 * (belief_vector[:, 0, 5] - belief_vector[:, 0, :5].max(dim=1)[0]))
            small_no_treat = torch.sigmoid(50 * (belief_vector[:, 1, 5] - belief_vector[:, 1, :5].max(dim=1)[0]))

            large_choice = F.softmax(belief_vector[:, 0, :5] * 50, dim=1)
            small_choice = F.softmax(belief_vector[:, 1, :5] * 50, dim=1)

            both_no_treat = large_no_treat * small_no_treat
            large_exists = 1 - large_no_treat
            small_exists = 1 - small_no_treat

            return both_no_treat.unsqueeze(1) * default_choice + large_exists.unsqueeze(1) * large_choice + (1 - large_exists).unsqueeze(1) * small_exists.unsqueeze(1) * small_choice
        else:
            large_choice = F.softmax(belief_vector[:, 0, :5] * 20, dim=1)
            small_choice = F.softmax(belief_vector[:, 1, :5] * 20, dim=1)
            conflict = torch.sum(large_choice * dominant_decision, dim=1, keepdim=True)
            conflict = torch.sigmoid(20 * (conflict - 0.5))
            return (1 - conflict) * large_choice + conflict * small_choice
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
        presence_weight = torch.sigmoid(50 * (opponent_presence - 0.5))
        return presence_weight * sub_decision + (1 - presence_weight) * greedy_decision


class AblationArchitecture(nn.Module):
    def __init__(self, module_configs: Dict[str, bool]):
        super().__init__()
        self.kwargs = {'module_configs': module_configs}
        self.kwargs['batch_size'] = 128
        self.perception = PerceptionModule(use_neural=module_configs.get('perception', True)) # get from dict, but if it doesn't exist return true
        self.my_belief = BeliefModule(use_neural=module_configs.get('my_belief', True))
        self.op_belief = BeliefModule(use_neural=module_configs.get('op_belief', True)) if not module_configs['shared_belief'] else self.my_belief
        self.my_greedy_decision = DecisionModule(is_subordinate=False, use_neural=module_configs.get('my_greedy_decision', True))
        self.op_greedy_decision = DecisionModule(is_subordinate=False, use_neural=module_configs.get('op_greedy_decision', True)) if not module_configs['shared_decision'] else self.my_greedy_decision
        self.sub_decision = DecisionModule(is_subordinate=True, use_neural=module_configs.get('sub_decision', True)) if not module_configs['shared_decision'] else self.my_greedy_decision
        self.final_output = FinalOutputModule(use_neural=module_configs.get('final_output', True))


        self.perception_old = PerceptionModuleOld(use_neural=False)
        self.my_belief_old = BeliefModuleOld(use_neural=False)
        self.op_belief_old = BeliefModuleOld(use_neural=False)
        self.my_greedy_decision_old = DecisionModuleOld(is_subordinate=False, use_neural=False)
        self.op_greedy_decision_old = DecisionModuleOld(is_subordinate=False, use_neural=False)
        self.sub_decision_old = DecisionModuleOld(is_subordinate=True, use_neural=False)
        self.final_output_old = FinalOutputModuleOld(use_neural=False)

    def compare_tensors(self, name: str, new_tensor: torch.Tensor, old_tensor: torch.Tensor, threshold: float = 0.1, **inputs):
        if isinstance(new_tensor, dict):
            for k in new_tensor.keys():
                self.compare_tensors(f"{name}[{k}]", new_tensor[k], old_tensor[k], threshold, **inputs)
            return

        if torch.isnan(new_tensor).any():
            print(f"\nNANs detected in {name}:")
            for i in range(new_tensor.shape[0]):
                if torch.isnan(new_tensor[i]).any():
                    print(f"Batch {i} contains NANs")
                    print("Inputs:")
                    for input_name, input_tensor in inputs.items():
                        if isinstance(input_tensor, dict):
                            print(f"{input_name}:")
                            for k, v in input_tensor.items():
                                print(f"  {k}:")
                                print(v[i])
                        else:
                            print(f"{input_name}:")
                            print(input_tensor[i])
                    print("Output with NANs:")
                    print(new_tensor[i])
                    break
            return

        # Iterate through batch dimension
        for i in range(new_tensor.shape[0]):
            # Compare decisions instead of raw values
            new_decision = torch.argmax(new_tensor[i].float(), dim=-1) if len(new_tensor[i].shape) > 0 else new_tensor[i]
            old_decision = torch.argmax(old_tensor[i].float(), dim=-1) if len(old_tensor[i].shape) > 0 else old_tensor[i]

            if not torch.equal(new_decision, old_decision):
                print(f"\nDecision difference in {name} at batch {i}:")
                print("Inputs:")
                for input_name, input_tensor in inputs.items():
                    if isinstance(input_tensor, dict):
                        print(f"{input_name}:")
                        for k, v in input_tensor.items():
                            print(f"  {k}:")
                            print(v[i])
                            if len(v[i].shape) > 0:
                                print(f"  {k} decisions:")
                                print(torch.argmax(v[i], dim=-1))
                    else:
                        print(f"{input_name}:")
                        print(input_tensor[i])
                        if len(input_tensor[i].shape) > 0:
                            print(f"{input_name} decisions:")
                            print(torch.argmax(input_tensor[i], dim=-1))
                print("New output & decision:")
                print(new_tensor[i])
                print(new_decision)
                print("Old output & decision:")
                print(old_tensor[i])
                print(old_decision)
                break
    
    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor) -> torch.Tensor:
        perception_output = self.perception.forward(perceptual_field)
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device

        treats_visible = perception_output['treats_visible'].float()
        opponent_vision = perception_output['opponent_vision'].float()
        opponent_presence = perception_output['opponent_presence'].float()

        my_belief_vector = self.my_belief.forward(treats_visible, torch.ones(batch_size, 5, device=device, dtype=torch.float))
        op_belief_vector = self.op_belief.forward(treats_visible, opponent_vision)
        my_greedy_decision = self.my_greedy_decision.forward(my_belief_vector)
        op_greedy_decision = self.op_greedy_decision.forward(op_belief_vector)
        sub_decision = self.sub_decision.forward(my_belief_vector, op_greedy_decision)
        final_decision = self.final_output.forward(opponent_presence, my_greedy_decision, sub_decision)

        '''perception_output_old = self.perception_old(perceptual_field)
        treats_visible_old = perception_output_old['treats_visible'].float()
        opponent_vision_old = perception_output_old['opponent_vision'].float()
        opponent_presence_old = perception_output_old['opponent_presence'].float()

        my_belief_vector_old = self.my_belief_old(treats_visible_old, torch.ones(batch_size, 5, device=device, dtype=torch.float))
        op_belief_vector_old = self.op_belief_old(treats_visible_old, opponent_vision_old)

        my_greedy_decision_old = self.my_greedy_decision_old(my_belief_vector_old)
        op_greedy_decision_old = self.op_greedy_decision_old(op_belief_vector_old)
        sub_decision_old = self.sub_decision_old(my_belief_vector_old, op_greedy_decision_old)
        final_decision_old = self.final_output_old(opponent_presence_old, my_greedy_decision_old, sub_decision_old)

        self.compare_tensors("perception", perception_output, perception_output_old,
                             perceptual_field=perceptual_field)

        # For belief comparison:
        self.compare_tensors("my_belief", my_belief_vector, my_belief_vector_old,
                             treats_visible=treats_visible,
                             vision=torch.ones(batch_size, 5, device=device, dtype=torch.float))

        self.compare_tensors("op_belief", op_belief_vector, op_belief_vector_old,
                             treats_visible=treats_visible,
                             vision=opponent_vision)

        # For decision comparison:
        self.compare_tensors("my_greedy_decision", my_greedy_decision, my_greedy_decision_old,
                             belief_vector=my_belief_vector)

        self.compare_tensors("op_greedy_decision", op_greedy_decision, op_greedy_decision_old,
                             belief_vector=op_belief_vector)

        self.compare_tensors("sub_decision", sub_decision, sub_decision_old,
                             belief_vector=my_belief_vector,
                             dominant_decision=op_greedy_decision)

        # For final output comparison:
        self.compare_tensors("final_decision", final_decision, final_decision_old,
                             opponent_presence=opponent_presence,
                             greedy_decision=my_greedy_decision,
                             sub_decision=sub_decision)'''
        
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