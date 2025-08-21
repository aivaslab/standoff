import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from typing import Dict

# new version
class EndToEndModel(nn.Module):
    def __init__(self, arch='mlp', output_type='my_decision'):
        super().__init__()
        self.arch = arch
        self.output_type = output_type
        raw_input_dim = 5 * 5 * 7 * 7
        processed_input_dim = 2 * 6 * 5 + 5 + 5  # treats(60) + vision(5) + presence_seq(5) = 70
        hidden = 128

        if output_type == 'op_belief':
            output_dim = 12
        elif output_type in ['op_decision', 'my_decision']:
            output_dim = 5

        if arch == 'mlp':
            self.raw_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(raw_input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
            self.processed_model = nn.Sequential(
                nn.Linear(processed_input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        elif arch == 'cnn':
            self.raw_model = nn.Sequential(
                nn.Conv3d(5, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
            self.processed_model = nn.Sequential(
                nn.Linear(processed_input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        elif arch == 'lstm128':
            self.raw_rnn = nn.LSTM(input_size=5*7*7, hidden_size=128, batch_first=True)
            self.processed_rnn = nn.LSTM(input_size=processed_input_dim//5, hidden_size=128, batch_first=True)
            self.head = nn.Linear(128, output_dim)
        elif arch == 'lstm32':
            self.raw_rnn = nn.LSTM(input_size=5*7*7, hidden_size=32, batch_first=True)
            self.processed_rnn = nn.LSTM(input_size=processed_input_dim//5, hidden_size=32, batch_first=True)
            self.head = nn.Linear(32, output_dim)
        elif arch == 'transformer128':
            self.raw_embed = nn.Linear(5*7*7, 128)
            self.processed_embed = nn.Linear(processed_input_dim//5, 128)
            encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
            self.head = nn.Linear(128, output_dim)
        elif arch == 'transformer32':
            self.raw_embed = nn.Linear(5*7*7, 32)
            self.processed_embed = nn.Linear(processed_input_dim//5, 32)
            encoder = nn.TransformerEncoderLayer(d_model=32, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
            self.head = nn.Linear(32, output_dim)
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, input_data):
        if len(input_data.shape) == 5:
            B, T, C, H, W = input_data.shape
            
            if self.arch == 'cnn':
                x = input_data.permute(0, 2, 1, 3, 4)
                output = self.raw_model(x)
            elif self.arch in ['mlp']:
                output = self.raw_model(input_data)
            elif self.arch in ['lstm128', 'lstm32', 'transformer128', 'transformer32']:
                x = input_data.view(B, T, -1)
                if 'transformer' in self.arch:
                    x = self.raw_embed(x)
                    x = self.transformer(x)
                else:
                    x = self.raw_rnn(x)[0]
                output = self.head(x[:, -1])
        else:
            B = input_data.shape[0]
            
            if self.arch in ['mlp', 'cnn']:
                output = self.processed_model(input_data)
            elif self.arch in ['lstm128', 'lstm32', 'transformer128', 'transformer32']:
                x = input_data.view(B, 5, -1)
                if 'transformer' in self.arch:
                    x = self.processed_embed(x)
                    x = self.transformer(x)
                else:
                    x = self.processed_rnn(x)[0]
                output = self.head(x[:, -1])

        if self.output_type == 'op_belief':
            output = output.view(B, 2, 6)
            return F.softmax(output, dim=-1)
        else:
            return F.softmax(output, dim=-1)


class BaseModule(nn.Module, ABC):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 50.0):
        super().__init__()
        self.use_neural = use_neural
        self.neural_network = self._create_neural_network() if use_neural else None
        self.random_prob = random_prob
        self.sigmoid_temp = sigmoid_temp
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


class FullEndToEndModule(BaseModule):
    def __init__(self, arch='mlp', **kwargs):
        self.arch = arch
        super().__init__(**kwargs)

    def _create_neural_network(self):
        return FullEndToEndModel(self.arch)

    def _hardcoded_forward(self, perceptual_field: torch.Tensor, *_):
        raise NotImplementedError()

    def _random_forward(self, perceptual_field: torch.Tensor, *_):
        batch_size = perceptual_field.shape[0]
        device = perceptual_field.device
        out = torch.zeros(batch_size, 5, device=device)
        out.scatter_(1, torch.randint(0, 5, (batch_size, 1), device=device), 1.0)
        return out


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
        return TreatPerceptionNetwork()

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


class RNNBeliefNetwork(nn.Module):
    def __init__(self, hidden_size=16, output_size=6):
        super().__init__()
        # Input size = 6 (treats) + 1 (vision) = 7 per timestep
        self.rnn = nn.RNN(input_size=7, hidden_size=hidden_size, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        print('using rnn beliefs')
        
    def forward(self, treats, vision):
        batch_size = treats.shape[0]
        
        vision = vision.reshape(batch_size, 5, 1)
    
        #treats = treats.permute(0, 2, 1)

        #print(vision.shape, treats.shape)
        
        combined_input = torch.cat([treats, vision], dim=2)  # Shape: [batch_size, 5, 7]
        
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=treats.device)
        
        _, hn = self.rnn(combined_input, h0)
        
        final_hidden = hn.squeeze(0)
        
        output = self.output_fc(final_hidden)
        output = F.softmax(output, dim=-1)
        
        return output


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

class NormalizedTSBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.input_norm = nn.BatchNorm1d(35)
        self.fc1 = nn.Linear(13, 12)
        self.fc2 = nn.Linear(12, 6)
        #self.fc3 = nn.Linear(16, 6)

    def forward(self, treats, vision, prev_beliefs):
        batch_size = treats.shape[0]

        x = torch.cat([treats.reshape(batch_size, -1), vision.reshape(batch_size, -1), prev_beliefs.reshape(batch_size, -1),], dim=1)
        #x = self.input_norm(x)
        x = F.leaky_relu_(self.fc1(x))
        x = self.fc2(x)
        #x = F.relu(self.fc3(x))
        x = x.view(batch_size, 6)
        #x = F.softmax(x, dim=-1)
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
        time_weights = self.time_weights[:, :visible_treats.size(1)]
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

class BeliefModulePerTimestep(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, uncertainty=0.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)
        self.uncertainty = uncertainty
        self.transition_counts = {}

    def _bin_beliefs(self, beliefs: torch.Tensor) -> list:
        quantized = torch.round(beliefs).long()
        return [tuple(row.tolist()) for row in quantized]

    def _create_neural_network(self) -> nn.Module:
        return NormalizedTSBeliefNetwork()

    def _count_transitions(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, new_beliefs: torch.Tensor):
        if not self.training:
            prev_state = self._bin_beliefs(prev_beliefs)
            new_state = self._bin_beliefs(new_beliefs)
            treat_state = self._bin_beliefs(visible_treats)
            vision_expanded = vision.expand(prev_beliefs.shape[0])
            vision_bins = (vision_expanded > 0.5).long()
            for i in range(prev_beliefs.shape[0]):
                transition = (prev_state[i], treat_state[i], vision_bins[i].item(), new_state[i])
                self.transition_counts[transition] = self.transition_counts.get(transition, 0) + 1
        
    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, t: int=0) -> torch.Tensor:
        time_weight = torch.exp(torch.tensor(t, dtype=torch.float32, device=visible_treats.device) * 2.0)
        update_mask = vision.unsqueeze(-1)

        weighted_obs = time_weight * update_mask * visible_treats

        new_beliefs = prev_beliefs + weighted_obs

        new_beliefs = new_beliefs / (new_beliefs.sum(dim=-1, keepdim=True) + 1e-10)

        return new_beliefs

    def _random_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        batch_size = visible_treats.shape[0]
        device = visible_treats.device

        beliefs = torch.zeros(batch_size, 6, device=device)
        random_indices = torch.randint(0, 6, (batch_size,), device=device)
        beliefs.scatter_(1, random_indices.unsqueeze(1), 1.0)

        return beliefs #F.softmax(beliefs, dim=-1)

    def forward(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, t: int=0) -> torch.Tensor:
        if self.use_neural:
            new_beliefs = self._neural_forward(visible_treats, vision, prev_beliefs)
        else:
            self.sigmoid_temp = 90.0
            new_beliefs = self._hardcoded_forward(visible_treats, vision, prev_beliefs, t)
        
        self._count_transitions(visible_treats, vision, prev_beliefs, new_beliefs)
        
        return new_beliefs

    def save_transition_table(self, filepath='transition_table.csv'):
        import pandas as pd
        
        rows = []
        for (prev_state, treat_state, vision, new_state), count in self.transition_counts.items():
            rows.append({
                'from_state': str(prev_state), 
                'treat_state': str(treat_state), 
                'vision': vision, 
                'to_state': str(new_state), 
                'count': count
            })
        
        df = pd.DataFrame(rows)
        
        df.to_csv(filepath, index=False)
        
        pivot_table = df.pivot_table(
            index=['from_state', 'treat_state', 'vision'], 
            columns='to_state', 
            values='count', 
            fill_value=0
        )
        pivot_filepath = filepath.replace('.csv', '_pivot.csv')
        pivot_table.to_csv(pivot_filepath)
        
        return df, pivot_table


class CombinerNetwork(nn.Module):
    def __init__(self, belief_dim=6, hidden_dim=12):
        super().__init__()

        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
        )

        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, belief_dim),
        )

    def forward(self, beliefs):

        belief_l = beliefs[..., 0, :]  # [batch_size, num_beliefs, belief_dim]
        belief_s = beliefs[..., 1, :]

        encoded_l = self.belief_encoder(belief_l)  # [batch_size, num_beliefs, hidden_dim]
        encoded_s = self.belief_encoder(belief_s)

        pooled_l = encoded_l.max(dim=1)[0]  # [batch_size, hidden_dim]
        pooled_s = encoded_s.max(dim=1)[0]

        combined_l = self.combiner(pooled_l)  # [batch_size, belief_dim]
        combined_s = self.combiner(pooled_s)

        combined_l = F.softmax(combined_l, dim=-1)  # [batch_size, belief_dim]
        combined_s = F.softmax(combined_s, dim=-1)

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
        
        self.process_opponent_perception = module_configs.get('opponent_perception', False)
        self.output_type = module_configs.get('output_type', 'my_decision')
        self.use_per_timestep_opponent = True
        self.store_per_timestep_beliefs = True

        sigmoid_temp = module_configs.get('sigmoid_temp', 50.0)

        self.register_buffer('null_decision', torch.zeros(self.kwargs['batch_size'], 5))
        self.register_buffer('null_presence', torch.zeros(self.kwargs['batch_size'], 1))

        print('module_configs', module_configs)
        print('random_probs', random_probs)

        print("Ablation Architecture")

        if random_probs is None:
            random_probs = {k: 0.0 for k in module_configs.keys()}

        if module_configs.get('end2end'):
            self.end2end_model = EndToEndModel(arch=module_configs['arch'], output_type=self.output_type)
        else:
            print("not e2e")
            self.end2end_model = None

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
        ) if not module_configs['shared_belief'] else self.my_belief) if not self.use_per_timestep_opponent else (BeliefModulePerTimestep(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0
        ) if not module_configs['shared_belief'] else self.my_belief)

        self.og_op_belief = BeliefModule(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0
        )

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

    def get_module_dict(self):
        return {
            'treat_perception_my': self.treat_perception_my,
            'treat_perception_op': self.treat_perception_op,
            'vision_perception_my': self.vision_perception_my,
            'vision_perception_op': self.vision_perception_op,
            'presence_perception_my': self.presence_perception_my,
            'presence_perception_op': self.presence_perception_op,
            'my_belief': self.my_belief,
            'op_belief': self.op_belief,
            'my_decision': self.my_decision,
            'op_decision': self.op_decision,
            'my_combiner': self.my_combiner,
            'op_combiner': self.op_combiner,
            'end2end_model': self.end2end_model
        }
    
    def get_trainable_modules(self):
        trainable = []
        for name, module in self.get_module_dict().items():
            if any(p.requires_grad for p in module.parameters()):
                trainable.append(name, module)
        return trainable
    
    def get_neural_modules(self):
        neural = []
        for name, module in self.get_module_dict().items():
            if module.use_neural:
                neural.append(name)
        return neural
    
    def freeze_modules(self, module_names):
        module_dict = self.get_module_dict()
        for name in module_names:
            if name in module_dict:
                for param in module_dict[name].parameters():
                    param.requires_grad = False
    
    def unfreeze_modules(self, module_names):
        module_dict = self.get_module_dict()
        for name in module_names:
            if name in module_dict:
                for param in module_dict[name].parameters():
                    param.requires_grad = True
    
    def set_module_training_state(self, trainable_modules, frozen_modules):
        all_modules = list(self.get_module_dict().keys())
        self.freeze_modules(all_modules)
        self.unfreeze_modules(trainable_modules)
        self.freeze_modules(frozen_modules)


    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor) -> torch.Tensor:
        device = perceptual_field.device
        batch_size = perceptual_field.shape[0]

        treats_op = self.treat_perception_op(perceptual_field)
        treats_l_op, treats_s_op = treats_op[:, :, 0:2].unbind(2)
        opponent_vision = self.vision_perception_op(perceptual_field, is_p1=0)
        opponent_presence = self.presence_perception_op(perceptual_field, is_p1=0)
        
        if self.detach_treat:
            treats_l_op = treats_l_op.detach()
            treats_s_op = treats_s_op.detach()
            opponent_vision = opponent_vision.detach()
            opponent_presence = opponent_presence.detach()

        if self.end2end_model is not None and self.output_type == 'my_decision':
            if self.process_opponent_perception:
                if opponent_presence.shape[1] == 1:
                    opponent_presence_seq = opponent_presence.repeat(1, 5)
                else:
                    opponent_presence_seq = opponent_presence
                end2end_input = torch.cat([
                    treats_op.flatten(start_dim=1),
                    opponent_vision.flatten(start_dim=1),
                    opponent_presence_seq.flatten(start_dim=1)
                ], dim=1)
            else:
                end2end_input = perceptual_field
            return {'my_decision': self.end2end_model(end2end_input)}

        if self.end2end_model is not None and self.output_type == 'op_belief':
            if self.process_opponent_perception:
                if opponent_presence.shape[1] == 1:
                    opponent_presence_seq = opponent_presence.repeat(1, 5)
                else:
                    opponent_presence_seq = opponent_presence
                end2end_input = torch.cat([
                    treats_op.flatten(start_dim=1),
                    opponent_vision.flatten(start_dim=1),
                    opponent_presence_seq.flatten(start_dim=1)
                ], dim=1)
            else:
                end2end_input = perceptual_field
            op_belief_vector = self.end2end_model(end2end_input)
        else:
            if self.use_per_timestep_opponent:
                op_belief_l = torch.zeros(batch_size, 6, device=device)
                op_belief_l[:, 5] = 1.0

                op_belief_s = torch.zeros(batch_size, 6, device=device) 
                op_belief_s[:, 5] = 1.0

                if self.store_per_timestep_beliefs:
                    op_beliefs_l_timesteps = []
                    op_beliefs_s_timesteps = []
                
                for t in range(5):
                    #op_belief_l = self.op_belief.forward(treats_l_op[:, :t+1], opponent_vision[:, :t+1])
                    #op_belief_s = self.op_belief.forward(treats_s_op[:, :t+1], opponent_vision[:, :t+1])
                    op_belief_l = self.op_belief.forward(treats_l_op[:, t], opponent_vision[:, t], op_belief_l, t)
                    op_belief_s = self.op_belief.forward(treats_s_op[:, t], opponent_vision[:, t], op_belief_s, t)

                    if self.store_per_timestep_beliefs:
                        op_beliefs_l_timesteps.append(op_belief_l.clone())
                        op_beliefs_s_timesteps.append(op_belief_s.clone())

                stacked_op_beliefs = torch.stack([
                    torch.stack(op_beliefs_l_timesteps, dim=1), 
                    torch.stack(op_beliefs_s_timesteps, dim=1)
                ], dim=1)

                #print(stacked_op_beliefs.shape)
                #op_belief_l = torch.softmax(op_belief_l, dim=-1)
                #op_belief_s = torch.softmax(op_belief_s, dim=-1)
                #following for debug
                og_op_belief_l = self.og_op_belief(treats_l_op * opponent_vision.unsqueeze(-1), opponent_vision)
                og_op_belief_s = self.og_op_belief(treats_s_op * opponent_vision.unsqueeze(-1), opponent_vision)
                og_op_beliefs = torch.stack([og_op_belief_l, og_op_belief_s], dim=1)

            else:
                self.op_belief.uncertainty = 0
                op_belief_l = self.op_belief(treats_l_op * opponent_vision.unsqueeze(-1), opponent_vision)
                op_belief_s = self.op_belief(treats_s_op * opponent_vision.unsqueeze(-1), opponent_vision)

            if self.detach_belief:
                op_belief_l = op_belief_l.detach()
                op_belief_s = op_belief_s.detach()

            op_beliefs = torch.stack([op_belief_l, op_belief_s], dim=1)
            op_belief_vector = self.op_combiner(op_beliefs.unsqueeze(1)) if self.use_combiner else op_beliefs

            if self.detach_combiner:
                op_belief_vector = op_belief_vector.detach()

        if self.end2end_model is not None and self.output_type == 'op_decision':
            if self.process_opponent_perception:
                if opponent_presence.shape[1] == 1:
                    opponent_presence_seq = opponent_presence.repeat(1, 5)
                else:
                    opponent_presence_seq = opponent_presence
                end2end_input = torch.cat([
                    treats_op.flatten(start_dim=1),
                    opponent_vision.flatten(start_dim=1),
                    opponent_presence_seq.flatten(start_dim=1)
                ], dim=1)
            else:
                end2end_input = perceptual_field
            op_decision = self.end2end_model(end2end_input)
        else:
            op_decision = self.op_decision(op_belief_vector, self.null_decision[:batch_size].to(device), self.null_presence[:batch_size].to(device))

            if self.detach_decision:
                op_decision = op_decision.detach()

        masked_visions = (torch.rand(batch_size, self.num_visions, 5, device=device) <= self.vision_prob).float()

        self.my_belief.uncertainty = 0.3
        beliefs_list = []
        treats_my = None
        my_vision = None
        for i in range(self.num_visions):
            masked_vision = masked_visions[:, i]
            masked_perceptual_field = perceptual_field * masked_vision.view(batch_size, 5, 1, 1, 1)
            my_vision = self.vision_perception_my(masked_perceptual_field, is_p1=1) if self.vision_prob < 1 else masked_vision
            treats_my = self.treat_perception_my(masked_perceptual_field)
            belief_l = self.my_belief.forward(treats_my[:,:,0] * my_vision.unsqueeze(-1), my_vision)
            belief_s = self.my_belief.forward(treats_my[:,:,1] * my_vision.unsqueeze(-1), my_vision)

            beliefs = torch.stack([belief_l, belief_s], dim=1)
            beliefs_list.append(beliefs)
        beliefs_tensor = torch.stack(beliefs_list, dim=1)
        my_belief_vector = self.my_combiner.forward(beliefs_tensor) if self.use_combiner else beliefs_tensor.squeeze(1)

        my_decision = self.my_decision.forward(my_belief_vector, op_decision, opponent_presence)

        result = {
            'treat_perception': treats_op,
            'vision_perception': opponent_vision,
            'vision_perception_my': my_vision,
            'presence_perception': opponent_presence,
            'my_belief': my_belief_vector,
            'og_op_belief': og_op_beliefs if self.store_per_timestep_beliefs else None,
            'op_belief': op_belief_vector,
            'my_decision': my_decision,
            'op_decision': op_decision
        }

        if self.store_per_timestep_beliefs and self.use_per_timestep_opponent:
            result['op_belief_t'] = stacked_op_beliefs

        return result