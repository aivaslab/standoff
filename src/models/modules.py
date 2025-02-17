import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from typing import Dict


class BaseModule(nn.Module, ABC):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 50.0):
        super().__init__()
        self.use_neural = use_neural
        self.neural_network = self._create_neural_network() if use_neural else None
        self.random_prob = random_prob
        self.sigmoid_temp = sigmoid_temp

        # print('random prob', self.random_prob)

    def _create_neural_network(self) -> nn.Module:
        pass

    def _hardcoded_forward(self, *args, **kwargs):
        pass

    def _neural_forward(self, *args, **kwargs):
        return self.neural_network(*args, **kwargs)
        pass

    def _random_forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        if self.use_neural:
            return self._neural_forward(*args, **kwargs)

        ## NOTE WE ARE SKIPPING RANDOM!!!!
        return self._hardcoded_forward(*args, **kwargs)

        batch_size = args[0].shape[0]
        device = args[0].device
        use_random = torch.rand(batch_size, device=device) < self.random_prob

        hardcoded = self._hardcoded_forward(*args, **kwargs)
        rand_output = self._random_forward(*args, **kwargs)
        # print(f"Fraction using random: {use_random.float().mean().item()}, {self.random_prob}")

        if isinstance(hardcoded, dict):
            result = {}
            for k in hardcoded.keys():
                mask_shape = [batch_size] + [1] * (hardcoded[k].dim() - 1)
                result[k] = torch.where(use_random.view(*mask_shape),
                                        rand_output[k],
                                        hardcoded[k])
            return result
        else:
            return torch.where(use_random.view(batch_size, *([1] * (hardcoded.dim() - 1))),
                               rand_output,
                               hardcoded)


# Perception module
# Inputs: The original perceptual field (5x5x7x7)
# Outputs: Visible treats (2x6 length vectors at each of 5 timesteps, normalized (position 5 is nothing)), opponent vision (scalar at each timestep), opponent presence (scalar overall) [56 total bits]
# Method: Find where treats are 1, where vision is shown, where opponent is.

class NormalizedPerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(5)
        self.attention = nn.Conv2d(5, 1, kernel_size=1)
        self.conv1 = nn.Conv2d(5, 24, kernel_size=3, padding=0)
        #self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, padding=0)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(31 * 3 * 3, 16)  # 2*6 + 1 = 13 outputs per timestep
        self.fc2 = nn.Linear(16, 13)
        self.fc_presence = nn.Linear(5 * 1 * 3 * 3, 1)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.zeros_(self.conv1.bias)

        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.zeros_(self.conv2.bias)

        #nn.init.normal_(self.fc.weight, std=0.01)
        #nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x):
        batch_size = x.shape[0]

        # From [batch, timestep, channel, height, width] to [batch * timestep, channel, height, width]
        x_reshaped = x.reshape(-1, *x.shape[2:])
        x_reshaped = self.input_bn(x_reshaped)
        #attn = torch.sigmoid(self.attention(x_reshaped))
        #x_reshaped = x_reshaped * attn

        h = F.relu(self.conv1(x_reshaped))
        h = F.relu(self.conv2(h))
        h = self.dropout(h)

        h_main = h[:, :-1]
        h_presence = h[:, -1:]

        h_presence = h_presence.reshape(batch_size, -1)
        presence = torch.sigmoid(5 * self.fc_presence(h_presence.reshape(batch_size, -1)))

        h = h_main.flatten(1)
        h = F.relu(self.fc(h))
        h = F.relu(self.fc2(h))

        treats = h[:, :12].view(-1, 2, 6)  # [batch * timestep, 2, 6]
        vision = h[:, 12]  # [batch * timestep]

        treats = F.softmax(treats, dim=-1)
        vision = torch.sigmoid(vision)

        treats = treats.view(batch_size, 5, 2, 6)  # [batch, timestep, 2, 6]
        vision = vision.view(batch_size, 5)  # [batch, timestep]

        return {
            'treats_l': treats[:, :, 0],
            'treats_s': treats[:, :, 1],
            'opponent_vision': vision,
            'opponent_presence': presence
        }


class PerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return NormalizedPerceptionNetwork()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        # channel 0 is agents
        # channel 1 is boxes... always 3?
        # channel 2 is treat1, at x=3 and y=1-6
        # channel 3 is treat2, at x=3 and y=1-6
        # channel 4 is walls

        treats = torch.sigmoid(self.sigmoid_temp * (perceptual_field[:, :, 2:4, 1:6, 3] - 0.5))  # [batch, time, 2, 5]
        treats = torch.flip(treats, dims=[3])  # Flip y dimension

        treat_sums = treats.sum(dim=3, keepdim=True)  # [batch, time, 2, 1]
        no_treats = torch.sigmoid(self.sigmoid_temp * (0.1 - treat_sums))

        treats = torch.cat([treats, no_treats], dim=3)  # [batch, time, 2, 6]

        treats_l = treats[:, :, 0]  # [batch, time, 6]
        treats_s = treats[:, :, 1]  # [batch, time, 6]

        opponent_vision = torch.sigmoid(self.sigmoid_temp * (torch.abs(perceptual_field[:, :, 4, 3, 2] - 1.0) - 0.5))
        opponent_presence = perceptual_field[:, 0, 0, 3, 0].unsqueeze(1)

        return {
            'treats_l': treats_l,
            'treats_s': treats_s,
            'opponent_vision': opponent_vision,
            'opponent_presence': opponent_presence
        }

    def _random_forward(self, perceptual_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device

        treats_visible = torch.rand(batch_size, 5, 2, 6, device=device)
        treats_visible = F.softmax(treats_visible, dim=-1)

        opponent_vision = torch.rand(batch_size, 5, device=device)

        opponent_presence = torch.rand(batch_size, 1, device=device)

        return {
            'treats_l': treats_visible[:, :, 0],
            'treats_s': treats_visible[:, :, 1],
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
        self.input_norm = nn.BatchNorm1d(35)
        self.fc1 = nn.Linear(35, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, treats, vision):
        batch_size = treats.shape[0]

        x = torch.cat([treats.reshape(batch_size, -1), vision.reshape(batch_size, -1)], dim=1)
        x = self.input_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(batch_size, 6)
        x = F.softmax(x, dim=-1)
        return x


class BeliefModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, uncertainty=0.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)
        self.uncertainty = uncertainty

    def _create_neural_network(self) -> nn.Module:
        return NormalizedBeliefNetwork()

    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        device = visible_treats.device

        treats = visible_treats[:, :]
        positions = treats[..., :5]

        has_treat = torch.sigmoid(self.sigmoid_temp * (positions.max(dim=-1)[0] - 0.5))  # [batch, 5]
        valid_observations = has_treat * vision

        time_weights = torch.exp(torch.arange(5, device=device) * 2.0).view(1, 5)
        time_weighted_valid_obs = time_weights * valid_observations
        time_weighted_uncertain = time_weights * (1 - vision)
        # there are 3 types of timesteps: seen with treats, seen without, and unseen

        weighted_positions = time_weighted_valid_obs.unsqueeze(-1) * positions + \
                             self.uncertainty * torch.ones_like(positions) * time_weighted_uncertain.unsqueeze(-1)

        # so we have weighted positions, including 0.2 for all unseen ones
        # now, if any position was unseen before the end, it should be reduced, or uncertainty should be ADDED to all others.

        position_beliefs = weighted_positions.sum(dim=1)
        position_beliefs = position_beliefs / (
                    (time_weighted_valid_obs + time_weighted_uncertain).sum(dim=1, keepdim=True) + 1e-10)

        ever_see_treat = torch.sigmoid(self.sigmoid_temp * (
                    valid_observations.max(dim=1)[0] + (1 - vision).sum(dim=1)[0] * self.uncertainty - 0.5))
        no_treat_belief = (1 - ever_see_treat).unsqueeze(-1)

        belief = torch.cat([position_beliefs, no_treat_belief], dim=1)
        return belief / belief.sum(dim=1, keepdim=True)

    def _random_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        batch_size = visible_treats.shape[0]
        device = visible_treats.device

        beliefs = torch.rand(batch_size, 2, 6, device=device)

        return F.softmax(beliefs, dim=-1)


class CombinerNetwork(nn.Module):
    def __init__(self, belief_dim=6, hidden_dim=16):
        super().__init__()

        self.belief_encoder = nn.Sequential(
            nn.BatchNorm1d(belief_dim),
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim)
        )

    def forward(self, beliefs):
        batch_size = beliefs.shape[0]

        if beliefs.shape[1] == 1:
            return beliefs.squeeze(1)

        belief_l = beliefs[..., 0, :]  # [batch_size, num_beliefs, belief_dim]
        belief_s = beliefs[..., 1, :]

        encoded_l = self.belief_encoder(belief_l)  # [batch_size, num_beliefs, hidden_dim]
        encoded_s = self.belief_encoder(belief_s)

        pooled_l = encoded_l.mean(dim=1)  # [batch_size, hidden_dim]
        pooled_s = encoded_s.mean(dim=1)

        combined_l = self.combiner(pooled_l)  # [batch_size, belief_dim]
        combined_s = self.combiner(pooled_s)

        #combined_l = F.softmax(combined_l, dim=-1)  # [batch_size, belief_dim]
        #combined_s = F.softmax(combined_s, dim=-1)

        position_probs_l = combined_l[:, :-1]  # [batch_size, 5]
        position_probs_s = combined_s[:, :-1]  # [batch_size, 5]

        overlap = position_probs_l * position_probs_s
        overlap_sum = overlap.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        penalty = 1 - overlap_sum

        position_probs_l = position_probs_l * penalty
        position_probs_s = position_probs_s * penalty

        final_l = torch.cat([position_probs_l, combined_l[:, -1:]], dim=-1)
        final_s = torch.cat([position_probs_s, combined_s[:, -1:]], dim=-1)

        final_l = final_l / final_l.sum(dim=-1, keepdim=True)
        final_s = final_s / final_s.sum(dim=-1, keepdim=True)

        return torch.stack([final_l, final_s], dim=1)


class CombinerModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return CombinerNetwork()

    def _random_forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        batch_size = beliefs.shape[0]
        device = beliefs.device

        random_beliefs = torch.rand(batch_size, 2, 6, device=device)
        return F.softmax(random_beliefs, dim=-1)

    def _hardcoded_forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(beliefs, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (batch_size, num_visions, 2)

        weights = F.softmax(-entropy * self.sigmoid_temp, dim=1)  # (batch_size, num_visions, 2)

        weighted_beliefs = beliefs * weights.unsqueeze(-1)  # (batch_size, num_visions, 2, 6)
        combined_beliefs = weighted_beliefs.sum(dim=1)  # (batch_size, 2, 6)

        return F.softmax(combined_beliefs, dim=-1)


# Greedy Decision module
# Inputs: Belief vector
# Output: Decision
# Method: Argmax of the large treat belief unless it is 5, else argmax of small treat belief unless it is 5, else 2

class DecisionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0,
                 sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        input_size = 18
        return nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Softmax(dim=-1)
        )

    def _neural_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None, dominant_present: torch.Tensor = None) -> torch.Tensor:
        batch_size = belief_vector.shape[0]

        #print(belief_vector.shape, dominant_decision.shape, dominant_present.shape)

        x = torch.cat([belief_vector.reshape(batch_size, -1), dominant_decision, dominant_present], dim=-1)

        return self.neural_network(x)


    def _hardcoded_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None, dominant_present: torch.Tensor = None) -> torch.Tensor:
        batch_size = belief_vector.shape[0]
        device = belief_vector.device

        default_choice = torch.zeros(batch_size, 5, device=device)
        default_choice[:, 2] = 1.0

        #large_no_treat = torch.sigmoid(self.sigmoid_temp * (belief_vector[:, 0, 5] - belief_vector[:, 0, :5].max(dim=1)[0]))
        #small_no_treat = torch.sigmoid(self.sigmoid_temp * (belief_vector[:, 1, 5] - belief_vector[:, 1, :5].max(dim=1)[0]))

        #large_treat_prob = belief_vector[:, 0, :5].sum(dim=1)  # Probability of any large treat
        large_no_treat = belief_vector[:, 0, 5]
        small_no_treat = belief_vector[:, 1, 5]

        #large_choice = F.softmax(belief_vector[:, 0, :5], dim=1)
        #small_choice = F.softmax(belief_vector[:, 1, :5], dim=1)

        large_choice = belief_vector[:, 0, :5]
        small_choice = belief_vector[:, 1, :5]

        both_no_treat = large_no_treat * small_no_treat
        large_exists = 1 - large_no_treat
        small_exists = 1 - small_no_treat

        greedy_decision = (
                both_no_treat.unsqueeze(1) * default_choice +
                large_exists.unsqueeze(1) * large_choice +
                large_no_treat.unsqueeze(1) * small_exists.unsqueeze(1) * small_choice
        )

        conflict = torch.sum(large_choice * dominant_decision, dim=1, keepdim=True)
        conflict = torch.sigmoid((conflict - 0.5))
        subordinate_decision = (1 - conflict) * large_choice + conflict * small_choice

        return dominant_present * subordinate_decision + (1 - dominant_present) * greedy_decision

    def _random_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None, dominant_present: torch.Tensor = None) -> torch.Tensor:
        decisions = torch.rand(belief_vector.shape[0], 5, device=belief_vector.device)
        decisions = F.softmax(decisions, dim=-1)
        return decisions



class AblationArchitecture(nn.Module):
    def __init__(self, module_configs: Dict[str, bool], random_probs: Dict[str, float] = None):
        super().__init__()
        self.kwargs = {'module_configs': module_configs}
        self.kwargs['batch_size'] = 128
        self.vision_prob = module_configs.get('vision_prob', 1.0)
        self.num_visions = module_configs.get('num_beliefs', 1)

        sigmoid_temp = module_configs.get('sigmoid_temp', 50.0)

        self.null_decision = torch.zeros(self.kwargs['batch_size'], 5,)
        self.null_presence = torch.zeros(self.kwargs['batch_size'], 1,)

        print('temperature:', sigmoid_temp)

        if random_probs is None:
            random_probs = {k: 0.0 for k in module_configs.keys()}


        self.perception = PerceptionModule(
            use_neural=module_configs['perception'],
            random_prob=random_probs['perception'],
            sigmoid_temp=sigmoid_temp
        )

        self.my_belief = BeliefModule(
            use_neural=module_configs['my_belief'],
            random_prob=random_probs['my_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.3
        )

        self.combiner = CombinerModule(
            use_neural=module_configs['combiner'],
            random_prob=random_probs['combiner'],
            sigmoid_temp=sigmoid_temp
        )

        self.op_belief = (BeliefModule(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0
        ) if not module_configs['shared_belief'] else self.my_belief)


        self.my_decision = DecisionModule(
            use_neural=module_configs['my_decision'],
            random_prob=random_probs['my_decision'],
            sigmoid_temp=sigmoid_temp
        )

        self.op_decision = (DecisionModule(
            use_neural=module_configs['op_decision'],
            random_prob=random_probs['op_decision'],
            sigmoid_temp=sigmoid_temp
        ) if not module_configs['shared_decision'] else self.my_decision)


    def compare_tensors(self, name: str, new_tensor: torch.Tensor, old_tensor: torch.Tensor, threshold: float = 0.1,
                        **inputs):
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

        for i in range(new_tensor.shape[0]):
            new_decision = torch.argmax(new_tensor[i].float(), dim=-1) if len(new_tensor[i].shape) > 0 else new_tensor[
                i]
            old_decision = torch.argmax(old_tensor[i].float(), dim=-1) if len(old_tensor[i].shape) > 0 else old_tensor[
                i]

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

        treats_l = perception_output['treats_l'].float()
        treats_s = perception_output['treats_s'].float()

        opponent_vision = perception_output['opponent_vision'].float()
        opponent_presence = perception_output['opponent_presence'].float()

        op_belief_l = self.op_belief.forward(treats_l, opponent_vision)
        op_belief_s = self.op_belief.forward(treats_s, opponent_vision)
        op_beliefs = torch.stack([op_belief_l, op_belief_s], dim=1)

        beliefs_list = []
        for i in range(self.num_visions):
            masked_vision = (torch.rand(batch_size, 5, device=device) <= self.vision_prob).float()

            belief_l = self.my_belief.forward(treats_l * masked_vision.unsqueeze(-1), masked_vision)
            belief_s = self.my_belief.forward(treats_s * masked_vision.unsqueeze(-1), masked_vision)

            beliefs = torch.stack([belief_l, belief_s], dim=1)  # [batch, 2, 6]
            beliefs_list.append(beliefs)

        beliefs_tensor = torch.stack(beliefs_list, dim=1)

        my_belief_vector = self.combiner.forward(beliefs_tensor)
        op_decision = self.op_decision.forward(op_beliefs, self.null_decision[:batch_size].to(device), self.null_presence[:batch_size].to(device))
        my_decision = self.my_decision.forward(my_belief_vector, op_decision, opponent_presence)

        return my_decision
