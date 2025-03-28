import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from typing import Dict

# new version


class BaseModule(nn.Module, ABC):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 50.0):
        super().__init__()
        self.use_neural = use_neural
        self.neural_network = self._create_neural_network() if use_neural else None
        self.random_prob = random_prob
        self.sigmoid_temp = sigmoid_temp
        #print(str(self), self.sigmoid_temp)

        # print('random prob', self.random_prob)
        if use_neural and self.neural_network is not None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.neural_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        self.sigmoid_temp = 90.0 # this is stupid but sometimes they are
        return self._hardcoded_forward(*args, **kwargs)

        '''batch_size = args[0].shape[0]
        device = args[0].device
        use_random = torch.rand(batch_size, device=device) <= self.random_prob

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
                               hardcoded)'''


# Perception module
# Inputs: The original perceptual field (5x5x7x7)
# Outputs: Visible treats (2x6 length vectors at each of 5 timesteps, normalized (position 5 is nothing)), opponent vision (scalar at each timestep), opponent presence (scalar overall) [56 total bits]
# Method: Find where treats are 1, where vision is shown, where opponent is.

class TreatPerceptionNetworkOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.treat_detector = nn.Sequential(
            nn.Linear(5 * 1, 6),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_timesteps = x.shape[1]

        treat1_input = x[:, :, 2, 1:6, 3].reshape(batch_size * num_timesteps, 5 * 1)
        treat1 = self.treat_detector(treat1_input).view(batch_size, num_timesteps, 6)

        treat2_input = x[:, :, 3, 1:6, 3].reshape(batch_size * num_timesteps, 5 * 1)
        treat2 = self.treat_detector(treat2_input).view(batch_size, num_timesteps, 6)
        
        treats = torch.stack([treat1, treat2], dim=2)
        treats = F.softmax(treats, dim=-1)

        return treats

class TreatPerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.treat_detector = nn.Sequential(
            nn.Linear(5 * 5, 5 * 6),
            #nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_timesteps = x.shape[1]

        treat1_input = x[:, :, 2, 1:6, 3].reshape(batch_size, 5 * 5)
        treat1 = self.treat_detector(treat1_input).view(batch_size, 5, 6)

        treat2_input = x[:, :, 3, 1:6, 3].reshape(batch_size, 5 * 5)
        treat2 = self.treat_detector(treat2_input).view(batch_size, 5, 6)
        
        treats = torch.stack([treat1, treat2], dim=2)
        treats = F.softmax(treats, dim=-1)

        return treats

class TreatPerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return TreatPerceptionNetworkOld()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        # channel 2 is treat1, channel 3 is treat2, at x=3 and y=1-6
        logits = self.sigmoid_temp * (perceptual_field[:, :, 2:4, 1:6, 3] - 0.5)
        logits = torch.flip(logits, dims=[3])
        logit_sums = logits.exp().sum(dim=3, keepdim=True)
        no_treats_logits = self.sigmoid_temp * (0.1 - logit_sums)
        combined_logits = torch.cat([logits, no_treats_logits], dim=3)
        treats = F.softmax(combined_logits, dim=3)
        return treats

    def _random_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        batch_size = perceptual_field.shape[0]
        num_timesteps = perceptual_field.shape[1]
        device = perceptual_field.device
        
        treats = torch.zeros(batch_size, num_timesteps, 2, 6, device=device)
        
        random_positions_t1 = torch.randint(0, 6, (batch_size, num_timesteps, 1), device=device)
        random_positions_t2 = torch.randint(0, 6, (batch_size, num_timesteps, 1), device=device)
        
        treats.view(batch_size, num_timesteps, 2*6).scatter_(
            2, 
            torch.cat([
                random_positions_t1 + 0*6, 
                random_positions_t2 + 1*6 
            ], dim=2),
            1.0
        )
        
        # Reshape back
        treats = treats.view(batch_size, num_timesteps, 2, 6)
        
        return treats
        

class VisionPerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_detector = nn.Sequential(
            nn.Linear(5, 5),
        )

    def forward(self, x, is_p1=0):
        batch_size, timesteps = x.shape[:2]
        x = x[:, :, 4, 3, 2+3*is_p1]  # Get vision spot
        x = x.reshape(batch_size, 5)  
        x = torch.sigmoid_(self.vision_detector(x))
        return x.view(batch_size, timesteps)

class VisionPerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return VisionPerceptionNetwork()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor, is_p1=0) -> torch.Tensor:
        # channel 4 position (3,2) indicates vision
        return torch.sigmoid_(20 * (torch.abs(perceptual_field[:, :, 4, 3, 2+3*is_p1] - 1.0) - 0.5))

    def _random_forward(self, perceptual_field: torch.Tensor, is_p1) -> torch.Tensor:
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device
        return torch.randint(0, 2, (batch_size, 5), device=device).float() #torch.rand(batch_size, 5, device=device)


class PresencePerceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.presence_detector = nn.Sequential(
            nn.Linear(1, 1),  # Just looking at one 7x7 grid to output presence
            #nn.ReLU()
        )

    def forward(self, x, is_p1=0):
        batch_size = x.shape[0]
        x = x[:, 0, 0, 3, 0+6*is_p1]  # Get channel 0 at first timestep
        x = x.reshape(batch_size, 1)  # Flatten spatial dimensions
        x = torch.sigmoid_(5*self.presence_detector(x))
        return x

class PresencePerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        return PresencePerceptionNetwork()

    def _hardcoded_forward(self, perceptual_field: torch.Tensor, is_p1) -> torch.Tensor:
        # channel 0 position (3,0) at time 0 indicates presence
        #if is_p1:
        #    print(perceptual_field[:, 0, 0, 3, 0+6*is_p1][:20])
        return perceptual_field[:, 0, 0, 3, 0+6*is_p1].unsqueeze(1)

    def _random_forward(self, perceptual_field: torch.Tensor, is_p1) -> torch.Tensor:
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device
        return torch.randint(0, 2, (batch_size, 1), device=device).float() #torch.rand(batch_size, 1, device=device)


# Belief module
# Inputs: Visible Treats 6x5, Vision 1x5 (1s if subject)
# Outputs: Belief vector at last timestep (2x6, normalized)
# Method: Start at the last timestep, and step backwards until we find the last visible instance of each treat

class NormalizedBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.input_norm = nn.BatchNorm1d(35)
        self.fc1 = nn.Linear(35, 16)
        self.fc2 = nn.Linear(16, 6)
        #self.fc3 = nn.Linear(16, 6)

    def forward(self, treats, vision):
        batch_size = treats.shape[0]

        x = torch.cat([treats.reshape(batch_size, -1), vision.reshape(batch_size, -1)], dim=1)
        #x = self.input_norm(x)
        x = F.leaky_relu_(self.fc1(x))
        x = self.fc2(x)
        #x = F.relu(self.fc3(x))
        x = x.view(batch_size, 6)
        x = F.softmax(x, dim=-1)
        #x = torch.sigmoid(x)
        return x


class BeliefModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, uncertainty=0.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)
        self.uncertainty = uncertainty
        self.register_buffer('time_weights', torch.exp(torch.arange(5) * 2).view(1, 5))

    def _create_neural_network(self) -> nn.Module:
        return NormalizedBeliefNetwork()
        

    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        time_weights = self.time_weights.to(visible_treats.device)
        uncertainty = self.uncertainty
        sigmoid_temp = self.sigmoid_temp

        treats = visible_treats[:, :]
        positions = treats[..., :5]
        has_treat = positions.max(dim=-1)[0]
        valid_observations = has_treat * vision
        time_weighted_valid_obs = time_weights * valid_observations
        time_weighted_uncertain = time_weights * (1.0 - vision)
        weighted_positions = time_weighted_valid_obs.unsqueeze(-1) * positions + uncertainty * torch.ones_like(positions, dtype=torch.float32) * time_weighted_uncertain.unsqueeze(-1)
        position_beliefs = weighted_positions.sum(dim=1) / ((time_weighted_valid_obs + time_weighted_uncertain).sum(dim=1, keepdim=True) + 1e-10)
        never_see_treat = 1 - torch.sigmoid(sigmoid_temp * (valid_observations.max(dim=1)[0] + (1.0 - vision).sum(dim=1)[0] * uncertainty - 0.5))
        belief = torch.cat([position_beliefs, never_see_treat.unsqueeze(-1)], dim=1)
        return belief / belief.sum(dim=1, keepdim=True)

    def _random_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        batch_size = visible_treats.shape[0]
        device = visible_treats.device

        beliefs = torch.zeros(batch_size, 6, device=device)
        random_indices = torch.randint(0, 6, (batch_size,), device=device)
        beliefs.scatter_(1, random_indices.unsqueeze(1), 1.0)

        return beliefs #F.softmax(beliefs, dim=-1)


class CombinerNetwork(nn.Module):
    def __init__(self, belief_dim=6, hidden_dim=12):
        super().__init__()

        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
        )

        self.combiner = nn.Sequential(
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
            #nn.Sigmoid(),
        )

    def forward(self, beliefs):

        belief_l = beliefs[..., 0, :]  # [batch_size, num_beliefs, belief_dim]
        belief_s = beliefs[..., 1, :]

        encoded_l = self.belief_encoder(belief_l)  # [batch_size, num_beliefs, hidden_dim]
        encoded_s = self.belief_encoder(belief_s)
  # [batch_size, num_beliefs, 1]
        #weighted_encoded_l = encoded_l * vision_weights
        #weighted_encoded_s = encoded_s * vision_weights

        pooled_l = encoded_l.max(dim=1)[0]  # [batch_size, hidden_dim]
        pooled_s = encoded_s.max(dim=1)[0]

        combined_l = self.combiner(pooled_l)  # [batch_size, belief_dim]
        combined_s = self.combiner(pooled_s)

        #overlap = combined_l * combined_s
        #overlap_sum = overlap.max(dim=-1, keepdim=True)[0]  # [batch_size, 1]
        #penalty = 1 - 0.1*overlap_sum

        #combined_l = combined_l * penalty
        #combined_s = combined_s * penalty

        combined_l = F.softmax(combined_l, dim=-1)  # [batch_size, belief_dim]
        combined_s = F.softmax(combined_s, dim=-1)

        '''
        #position_probs_l = combined_l[:, :-1]  # [batch_size, 5]
        #position_probs_s = combined_s[:, :-1]  # [batch_size, 5]
        position_probs_l = combined_l
        position_probs_s = combined_s

        overlap = position_probs_l * position_probs_s
        overlap_sum = overlap.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        penalty = 1 - overlap_sum

        position_probs_l = position_probs_l * penalty
        position_probs_s = position_probs_s * penalty

        #final_l = torch.cat([position_probs_l, combined_l[:, -1:]], dim=-1)
        #final_s = torch.cat([position_probs_s, combined_s[:, -1:]], dim=-1)
        final_l = position_probs_l
        final_s = position_probs_s

        final_l = final_l / final_l.sum(dim=-1, keepdim=True)
        final_s = final_s / final_s.sum(dim=-1, keepdim=True)'''

        return torch.stack([combined_l, combined_s], dim=1)


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
        #probs = beliefs
        weights = 1-(beliefs * torch.log(beliefs + 1e-10)).sum(dim=-1)  # (batch_size, num_visions, 2)

        #weights = vision_sums / vision_sums.sum(dim=1, keepdim=True).clamp(min=1e-10)
        weights = weights.unsqueeze(-1)
        #weights = 1-entropy  # (batch_size, num_visions, 2)

        weighted_beliefs = beliefs*weights  # (batch_size, num_visions, 2, 6)
        combined_beliefs = weighted_beliefs.max(dim=1)[0]  # (batch_size, 2, 6)

        return torch.sigmoid(5*(combined_beliefs-0.5))


# Greedy Decision module
# Inputs: Belief vector
# Output: Decision
# Method: Argmax of the large treat belief unless it is 5, else argmax of small treat belief unless it is 5, else 2

class DecisionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        input_size = 18
        return nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Sigmoid()
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

        #large_no_treat = torch.sigmoid((belief_vector[:, 0, 5] - belief_vector[:, 0, :5].max(dim=1)[0]))
        #small_no_treat = torch.sigmoid((belief_vector[:, 1, 5] - belief_vector[:, 1, :5].max(dim=1)[0]))

        large_no_treat = belief_vector[:, 0, 5]
        small_no_treat = belief_vector[:, 1, 5]

        large_choice = belief_vector[:, 0, :5]#F.softmax(belief_vector[:, 0, :5] * 4, dim=1)
        small_choice = belief_vector[:, 1, :5]#F.softmax(belief_vector[:, 1, :5] * 4, dim=1)

        both_no_treat = large_no_treat * small_no_treat
        large_exists = 1 - large_no_treat
        small_exists = 1 - small_no_treat

        greedy_decision = both_no_treat.unsqueeze(1) * default_choice + large_exists.unsqueeze(1) * large_choice + (1 - large_exists).unsqueeze(1) * small_exists.unsqueeze(1) * small_choice

        conflict = torch.max(large_choice * dominant_decision, dim=1, keepdim=True)[0] # either sum or max works here... but sum needs more sigmoids
        #conflict = torch.sigmoid((conflict - 0.5))
        subordinate_decision = (1 - conflict) * large_choice + conflict * small_choice

        return dominant_present * subordinate_decision + (1 - dominant_present) * greedy_decision

    def _random_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None, dominant_present: torch.Tensor = None) -> torch.Tensor:
        #decisions = torch.rand(belief_vector.shape[0], 5, device=belief_vector.device)
        #decisions = F.softmax(decisions, dim=-1)
        batch_size = belief_vector.shape[0]
        device = belief_vector.device
        decisions = torch.zeros(batch_size, 5, device=device)
        random_indices = torch.randint(0, 5, (batch_size,), device=device)
        decisions.scatter_(1, random_indices.unsqueeze(1), 1.0)
        return decisions



class AblationArchitecture(nn.Module):
    def __init__(self, module_configs: Dict[str, bool], random_probs: Dict[str, float] = None, batch_size=256):
        super().__init__()
        self.kwargs = {'module_configs': module_configs}
        self.kwargs['batch_size'] = batch_size
        self.vision_prob_base = module_configs.get('vision_prob', 1.0)
        self.vision_prob = self.vision_prob_base
        self.num_visions = module_configs.get('num_beliefs', 1)
        self.detach_treat = module_configs['shared_treat'] and module_configs['my_treat'] and module_configs['detach']
        self.detach_belief = module_configs['shared_belief'] and module_configs['my_belief'] and module_configs['detach']
        self.detach_decision = module_configs['shared_decision'] and module_configs['my_decision'] and module_configs['detach']
        self.detach_combiner = module_configs['shared_combiner'] and module_configs['combiner'] and module_configs['detach']
        self.use_combiner = module_configs['use_combiner']

        sigmoid_temp = module_configs.get('sigmoid_temp', 50.0)

        self.register_buffer('null_decision', torch.zeros(self.kwargs['batch_size'], 5))
        self.register_buffer('null_presence', torch.zeros(self.kwargs['batch_size'], 1))

        print('module_configs', module_configs)
        print('random_probs', random_probs)

        if random_probs is None:
            random_probs = {k: 0.0 for k in module_configs.keys()}

        self.treat_perception_my = TreatPerceptionModule(
            use_neural=module_configs['my_treat'],
            random_prob=random_probs['my_treat'],
            sigmoid_temp=sigmoid_temp
        )

        self.treat_perception_op = (TreatPerceptionModule(
            use_neural=module_configs['op_treat'],
            random_prob=random_probs['op_treat'],
            sigmoid_temp=sigmoid_temp
        ) if not module_configs['shared_treat'] else self.treat_perception_my)

        self.vision_perception_my = VisionPerceptionModule(
            use_neural=module_configs['vision_my'],
            random_prob=random_probs['vision_my'],
            sigmoid_temp=sigmoid_temp
        )

        self.vision_perception_op = (VisionPerceptionModule(
            use_neural=module_configs['vision_op'],
            random_prob=random_probs['vision_op'],
            sigmoid_temp=sigmoid_temp
        )  if not module_configs['shared_treat'] else self.vision_perception_my)

        self.presence_perception_my = PresencePerceptionModule(
            use_neural=module_configs['presence_my'],
            random_prob=random_probs['presence_my'],
            sigmoid_temp=sigmoid_temp
        )

        self.presence_perception_op = (PresencePerceptionModule(
            use_neural=module_configs['presence_op'],
            random_prob=random_probs['presence_op'],
            sigmoid_temp=sigmoid_temp
        ) if not module_configs['shared_treat'] else self.presence_perception_my)

        self.my_belief = BeliefModule(
            use_neural=module_configs['my_belief'],
            random_prob=random_probs['my_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.3 if not module_configs['shared_belief'] else 0.3
        )

        self.my_combiner = CombinerModule(
            use_neural=module_configs['combiner'],
            random_prob=random_probs['combiner'],
            sigmoid_temp=sigmoid_temp
        )

        self.op_combiner = (CombinerModule(
            use_neural=module_configs['combiner'],
            random_prob=random_probs['combiner'],
            sigmoid_temp=sigmoid_temp
        ) if not module_configs['shared_combiner'] else self.my_combiner)

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
        device = perceptual_field.device
        batch_size = perceptual_field.shape[0]

        treats_op = self.treat_perception_op(perceptual_field)
        treats_l_op, treats_s_op = treats_op[:,:,0:2].unbind(2)

        opponent_vision = self.vision_perception_op(perceptual_field, is_p1=0)
        opponent_presence = self.presence_perception_op(perceptual_field, is_p1=0)

        if self.detach_treat:
            treats_l_op = treats_l_op.detach()
            treats_s_op = treats_s_op.detach()
            opponent_vision = opponent_vision.detach()
            opponent_presence = opponent_presence.detach()


        self.op_belief.uncertainty = 0
        op_belief_l = self.op_belief.forward(treats_l_op * opponent_vision.unsqueeze(-1), opponent_vision)
        op_belief_s = self.op_belief.forward(treats_s_op * opponent_vision.unsqueeze(-1), opponent_vision)

        if self.detach_belief:
            op_belief_l = op_belief_l.detach()
            op_belief_s = op_belief_s.detach()
        op_beliefs = torch.stack([op_belief_l, op_belief_s], dim=1)

        op_belief_vector = self.op_combiner.forward(op_beliefs.unsqueeze(1)) if self.use_combiner else op_beliefs
        if self.detach_combiner:
            op_belief_vector = op_belief_vector.detach()

        #masked_visions = torch.sigmoid_(1000 * (self.vision_prob - torch.rand(batch_size, self.num_visions, 5, device=device) + 0.01))
        masked_visions = (torch.rand(batch_size, self.num_visions, 5, device=device) <= self.vision_prob).float()


        self.my_belief.uncertainty = 0.3
        beliefs_list = []
        for i in range(self.num_visions):
            masked_vision = masked_visions[:, i]
            # perceptual field is batch, timestep, channel, length, width
            masked_perceptual_field = perceptual_field * masked_vision.view(batch_size, 5, 1, 1, 1)
            my_vision = self.vision_perception_my(masked_perceptual_field, is_p1=1) if self.vision_prob < 1 else masked_vision
            treats = self.treat_perception_my(masked_perceptual_field)
            belief_l = self.my_belief.forward(treats[:,:,0] * my_vision.unsqueeze(-1), my_vision) # or masked_vision, since we know the thing?
            belief_s = self.my_belief.forward(treats[:,:,1] * my_vision.unsqueeze(-1), my_vision)

            beliefs = torch.stack([belief_l, belief_s], dim=1)  # [batch, 2, 6]
            beliefs_list.append(beliefs)
        beliefs_tensor = torch.stack(beliefs_list, dim=1)

        my_belief_vector = self.my_combiner.forward(beliefs_tensor) if self.use_combiner else beliefs_tensor.squeeze(1) 

        #my_presence = self.presence_perception_my(perceptual_field, 1).float()
        op_decision = self.op_decision.forward(op_belief_vector, self.null_decision[:batch_size].to(device), self.null_presence[:batch_size].to(device))
        if self.detach_decision:
            op_decision = op_decision.detach()
        my_decision = self.my_decision.forward(my_belief_vector, op_decision, opponent_presence)

        return {
            'treat_perception': treats,
            'vision_perception': opponent_vision,
            'presence_perception': opponent_presence,
            'my_belief': my_belief_vector,
            'op_belief': op_belief_vector,
            'my_decision': my_decision,
            'op_decision': op_decision
            }
