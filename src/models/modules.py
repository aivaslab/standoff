import numpy as np
import torch
import torch.nn as nn
from abc import ABC
import torch.nn.functional as F
from typing import Dict
from collections import defaultdict
import os
import pandas as pd



class EndToEndModel(nn.Module):
    def __init__(self, arch='mlp', output_type='my_decision', pad='', in_channels=3):
        super().__init__()
        self.arch = arch
        self.output_type = output_type
        self.transition_counts = defaultdict(float)
        self.register_buffer('transition_counts_tensor', torch.zeros(5, 2, 2, 6, 6, 6, dtype=torch.int64), persistent=False)
        self.T_in = 5
        self.T_pad = 5 if pad == '' or pad == False else 10
        self.register_buffer("causal_mask_full",torch.triu(torch.full((self.T_pad, self.T_pad), float("-inf")), diagonal=1))
        self.register_buffer("attn_mask", self.causal_mask_full[:self.T_pad, :self.T_pad])
        self.register_buffer("t_idx_base", torch.arange(self.T_in).view(1, -1))
        self.register_buffer("padded_zero", torch.zeros(self.T_pad, 32))
        #self.register_buffer("pos_embed", self._build_positional_encoding(self.T_pad, d_model=32))

        print('T_pad:', self.T_pad)
        raw_input_dim = self.T_pad * in_channels * 7 * 7
        processed_input_dim = 2 * 6 * self.T_in + self.T_in + self.T_in
        hidden = 32
        if output_type == 'op_belief':
            output_dim = 12
        elif output_type in ['op_decision', 'my_decision']:
            output_dim = 6
        elif output_type == 'multi':
            output_dim = 2*self.T_in*6 + 2*self.T_in*6 + self.T_in*6 + 5
        if arch == 'mlp':
            self.raw_model = nn.Sequential(nn.Flatten(), nn.Linear(raw_input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, output_dim))
            self.processed_model = nn.Sequential(nn.Linear(processed_input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, output_dim))
        elif arch == 'transformer32':
            self.raw_embed = nn.Linear(in_channels*7*7, 32)
            #self.pos_embed = nn.Parameter(torch.randn(1, self.T_pad, 32))
            self.processed_embed = nn.Linear(processed_input_dim//self.T_in, 32)
            encoder = nn.TransformerEncoderLayer(d_model=32, nhead=2, activation='gelu', dropout=0.2, batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
            #self.head = nn.Linear(32, output_dim)
            #self.head_op_bel   = nn.Linear(32, 2*6)
            #self.head_my_bel   = nn.Linear(32, 2*6)
            #self.head_op_dec_t = nn.Linear(32, 6)
            self.head_my_dec   = nn.Linear(32, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_positional_encoding(self, T_pad, d_model):
        pe = torch.zeros(1, T_pad, d_model)
        pos = torch.arange(T_pad, dtype=torch.float).unsqueeze(1)
        freqs = torch.linspace(0, 3.1415, d_model // 2)
        pe[0, :, 0::2] = torch.sin(pos * freqs)
        pe[0, :, 1::2] = torch.cos(pos * freqs)
        return pe

    def _pad_shift(self, x, T_in, T_pad, training):
        B = x.size(0)
        device = x.device
        feat = x.size(-1)

        shift = torch.randint(0, T_pad - T_in + 1, (B,), device=device) if training else torch.zeros(B, dtype=torch.long, device=device)

        t_idx = self.t_idx_base.to(device) + shift.view(B, 1)
        padded = self.padded_zero.to(device, dtype=x.dtype).expand(B, -1, -1).clone()
        b_idx = torch.arange(B, device=device).view(-1, 1).expand_as(t_idx)

        padded[b_idx, t_idx] = x
        last_idx = shift + T_in - 1
        return padded, t_idx, last_idx

    def forward(self, input_data):
        if len(input_data.shape) == 5:
            B, T, C, H, W = input_data.shape

            if self.arch == 'mlp':
                x_flat = input_data.view(B, -1)
                output = self.raw_model(x_flat)
                if self.output_type == 'multi':
                    return {
                        'op_belief_t': torch.zeros(B, 2, self.T_in, 6, device=input_data.device),
                        'my_belief_t': torch.zeros(B, 2, self.T_in, 6, device=input_data.device),
                        'op_decision_t': torch.zeros(B, self.T_in, 6, device=input_data.device),
                        'my_decision': F.softmax(output, dim=-1)
                    }
                elif self.output_type == 'my_decision' or self.output_type == 'op_decision':
                    return {'my_decision': output}
            else:
                x = input_data.view(B, T, -1)
                x = self.raw_embed(x)
                x_pad, win_idx, last_idx = self._pad_shift(x, T, self.T_pad, self.training)
                #x_pad = x_pad + self.pos_embed[:, :x_pad.size(1)]
                x_enc = self.transformer(x_pad, mask=self.attn_mask)
                #x_enc = self.transformer(x_pad)
                B2 = x_enc.size(0)
                b_idx = torch.arange(B2, device=x_enc.device).view(-1, 1).expand_as(win_idx)
                x_win = x_enc[b_idx, win_idx]                             # (B,T,d)
                #pooled = x_enc[torch.arange(B2, device=x_enc.device), last_idx]
                pooled = x_win.mean(dim=1)
                #op_b = self.head_op_bel(x_win).reshape(B, T, 2, 6).permute(0, 2, 1, 3)
                #my_b = self.head_my_bel(x_win).reshape(B, T, 2, 6).permute(0, 2, 1, 3)
                #op_dt = self.head_op_dec_t(x_win).view(B, T, 6)
                output = {'my_decision': self.head_my_dec(pooled)}
                #output = {'op_belief_t': op_b, 'my_belief_t': my_b, 'op_decision_t': op_dt, 'my_decision': self.head_my_dec(pooled)}
        elif False:
            B = input_data.shape[0]
            per_step = 14
            T_in = input_data.shape[1] // per_step
            if self.arch in ['mlp']:
                output = self.processed_model(input_data)
            elif self.arch in ['transformer128', 'transformer32']:
                x = input_data.view(B, T_in, -1)
                if 'transformer' in self.arch:
                    x = self.processed_embed(x)
                    x_pad, win_idx, last_idx = self._pad_shift(x, T_in, self.T_pad, self.training)
                    x_enc = self.transformer(x_pad, mask=self.attn_mask)
                    b_idx = torch.arange(B, device=x_enc.device).view(-1, 1).expand_as(win_idx)
                    x_win = x_enc[b_idx, win_idx]                             # (B,T_in,d)
                    pooled = x_enc[torch.arange(B, device=x_enc.device), last_idx]
                    #op_b = self.head_op_bel(x_win).view(B, T_in, 2, 6).permute(0, 2, 1, 3).contiguous()
                    #my_b = self.head_my_bel(x_win).view(B, T_in, 2, 6).permute(0, 2, 1, 3).contiguous()
                    #op_dt = self.head_op_dec_t(x_win).view(B, T_in, 6)
                    output = {'my_decision': self.head_my_dec(pooled)}
                    #output = {'op_belief_t': op_b, 'my_belief_t': my_b, 'op_decision_t': op_dt, 'my_decision': self.head_my_dec(pooled)}
                else:
                    x_pad, win_idx, last_idx = self._pad_shift(x, T, self.T_pad, self.training)
                    x_seq = self.raw_rnn(x_pad)[0]
                    B2 = x_seq.size(0)
                    b_idx = torch.arange(B2, device=x_seq.device).view(-1, 1).expand_as(win_idx)
                    x_win = x_seq[b_idx, win_idx]
                    pooled = x_seq[torch.arange(B2, device=x_seq.device), last_idx]
                    #op_b = self.head_op_bel(x_win).view(B, T, 2, 6).permute(0, 2, 1, 3).contiguous()
                    #my_b = self.head_my_bel(x_win).view(B, T, 2, 6).permute(0, 2, 1, 3).contiguous()
                    #op_dt = self.head_op_dec_t(x_win).view(B, T, 6)
                    output = {'my_decision': self.head_my_dec(pooled)}
                    #output = {'op_belief_t': op_b, 'my_belief_t': my_b, 'op_decision_t': op_dt, 'my_decision': self.head_my_dec(pooled)}
            if not self.training:
                ob = output['op_belief_t']            # [B,2,T,6]
                self._log_expected_transitions(input_data, ob)
        if self.output_type == 'op_belief':
            if isinstance(output, dict):
                out = output['op_belief_t'][:, :, -1].contiguous().view(output['op_belief_t'].size(0), 2, 6)
            else:
                out = output.view(output.size(0), 2, 6)
            return F.softmax(out, dim=-1)
        elif self.output_type == 'multi':
            return output
        else:
            if isinstance(output, dict):
                return output
            return F.softmax(output, dim=-1)

    @staticmethod
    def _one_hot_from_logits(logits):
        idx = logits.argmax(dim=-1, keepdim=True)
        oh = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return oh

    @staticmethod
    def _bin_vec_to_tuple(x_row):
        return tuple(torch.round(x_row).long().tolist())

    def _parse_flat_end2end_input(self, x_flat: torch.Tensor):
        B, D = x_flat.shape
        assert D % 14 == 0, f"flat input must be 14*T; got D={D}"
        T = D // 14
        xps = x_flat.view(B, T, 14)
        treats = xps[:, :, :12].view(B, T, 2, 6)   # [B,T,2,6]
        vision = xps[:, :, 12]                      # [B,T]
        presence = xps[:, :, 13]                    # [B,T]
        tL = treats[:, :, 0, :]                     # [B,T,6]
        tS = treats[:, :, 1, :]                     # [B,T,6]
        return T, tL, tS, vision, presence

    @torch.no_grad()
    def _log_expected_transitions(self, x_flat: torch.Tensor, ob_logits: torch.Tensor):
        B, D = x_flat.shape
        T, tL, tS, vis, pres = self._parse_flat_end2end_input(x_flat)
        cur = ob_logits.argmax(-1).long()
        prev = torch.cat([torch.full((B, 2, 1), 5, dtype=torch.long, device=cur.device), cur[:, :, :-1]], dim=2)
        kL = tL.argmax(-1).long()
        kS = tS.argmax(-1).long()
        v = (vis > 0.5).long()
        p_any = (pres > 0.5).any(dim=1).long()
        for s in (0, 1):
            prev_s = prev[:, s]
            cur_s  = cur[:, s]
            ks = kL if s == 0 else kS
            for t in range(1, T):
                m_base = p_any == 1
                if not m_base.any():
                    continue
                enc = (prev_s[m_base, t] * 36 + ks[m_base, t] * 6 + cur_s[m_base, t])
                v_t = v[m_base, t]
                for vv in (0, 1):
                    m = v_t == vv
                    if m.any():
                        bins = torch.bincount(enc[m], minlength=216).view(6, 6, 6)
                        self.transition_counts_tensor[t, s, vv].add_(bins)

    def save_transition_table(self, filepath='transition_table.csv'):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        tc = self.transition_counts_tensor.detach().cpu()
        eye = torch.eye(6, dtype=torch.long)
        nz = (tc > 0).nonzero(as_tuple=False)
        rows = []
        for t, s, v, i, k, j in nz:
            rows.append({
                'timestep': int(t.item()),
                'treat_size': int(s.item()),
                'from_state': str(tuple(eye[i].tolist())),
                'treat_state': str(tuple(eye[k].tolist())),
                'vision': int(v.item()),
                'to_state': str(tuple(eye[j].tolist())),
                'count': float(tc[t, s, v, i, k, j].item()),
            })
        df = pd.DataFrame(rows, columns=['timestep','treat_size','from_state','treat_state','vision','to_state','count'])
        df.to_csv(filepath, index=False)
        if len(df) == 0:
            return df, pd.DataFrame()
        pivot = df.pivot_table(index=['timestep','treat_size','from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
        pivot.to_csv(filepath.replace('.csv', '_pivot.csv'))
        for t in sorted(df['timestep'].unique()):
            dft = df[df['timestep'] == t]
            dft.to_csv(filepath.replace('.csv', f'_t{t}.csv'), index=False)
            pivt = dft.pivot_table(index=['treat_size','from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
            pivt.to_csv(filepath.replace('.csv', f'_t{t}_pivot.csv'))
            for s in (0, 1):
                dfts = dft[dft['treat_size'] == s]
                if len(dfts) == 0:
                    continue
                dfts.to_csv(filepath.replace('.csv', f'_t{t}_s{s}.csv'), index=False)
                pivts = dfts.pivot_table(index=['from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
                pivts.to_csv(filepath.replace('.csv', f'_t{t}_s{s}_pivot.csv'))
        return df, pivot


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

class SmallTransformer(nn.Module):
    def __init__(self, total_in_dim, total_out_dim, T=5, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.T = T
        self.input_dim = total_in_dim // T
        self.output_dim = total_out_dim // T if total_out_dim % T == 0 else total_out_dim
        self.embed = nn.Linear(self.input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, activation='gelu',
            batch_first=True, dropout=0.2
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, self.output_dim)
        self.register_buffer("causal_mask", torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        # x: [B, total_in_dim] - reshape to [B, T, input_dim]
        B = x.size(0)
        x = x.view(B, self.T, self.input_dim)
        x = self.embed(x)
        x = self.encoder(x, mask=self.causal_mask[:self.T, :self.T])
        out = self.head(x)
        return out.reshape(B, -1)

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
            nn.Linear(5 * 5, 2 * 5 * 6),
            #nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_timesteps = x.shape[1]

        treat_input = x[:, :, 1, 1:6, 3].reshape(batch_size, 5 * 5)
        treat = self.treat_detector(treat1_input).view(batch_size, 5, 2, 6)
        treats = F.softmax(treats, dim=-1)

        return treats

class TreatPerceptionModule(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, archx='mlp'):
        self.archx = archx
        super().__init__(use_neural, random_prob, sigmoid_temp)

    def _create_neural_network(self) -> nn.Module:
        if self.archx == "transformer":
            return SmallTransformer(245 * 5, 5 * 2 * 6, d_model=32)
        return TreatPerceptionNetwork()

    def _hardcoded_forward_2x(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        # channel 2 is treat1, channel 3 is treat2, at x=3 and y=1-6
        logits = self.sigmoid_temp * (perceptual_field[:, :, 2:4, 1:6, 3] - 0.5)
        logits = torch.flip(logits, dims=[3])
        logit_sums = logits.exp().sum(dim=3, keepdim=True)
        no_treats_logits = self.sigmoid_temp * (0.1 - logit_sums)
        combined_logits = torch.cat([logits, no_treats_logits], dim=3)
        treats = F.softmax(combined_logits, dim=3)
        return treats

    def _hardcoded_forward(self, perceptual_field: torch.Tensor) -> torch.Tensor:
        x = perceptual_field[:, :, 1, 1:6, 3]          # [B, T, 5]

        large_mask = torch.sigmoid(self.sigmoid_temp * (x - 1.5))
        small_mask = torch.sigmoid(self.sigmoid_temp * (1.5 - x)) * torch.sigmoid(self.sigmoid_temp * (x - 0.5))
        none_mask  = torch.sigmoid(self.sigmoid_temp * (0.5 - x)) 

        logits = self.sigmoid_temp * (torch.stack([large_mask, small_mask], dim=2) - 0.5)
        logits = torch.flip(logits, dims=[3])  

        logit_sums = logits.exp().sum(dim=3, keepdim=True)
        no_treat_logits = self.sigmoid_temp * (none_mask.unsqueeze(2) - logit_sums)
        combined_logits = torch.cat([logits, no_treat_logits], dim=3)
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
        x = x[:, :, 2, 3, 2+3*is_p1]  # Get vision spot
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
        return torch.sigmoid_(self.sigmoid_temp * (torch.abs(perceptual_field[:, :, 2, 3, 2+3*is_p1] - 1.0) - 0.5))

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
        x = torch.sigmoid_(25*self.presence_detector(x))
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
        return perceptual_field[:, :, 0, 3, 0+6*is_p1].amax(dim=1, keepdim=True)

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

class NormalizedBeliefTransformer(nn.Module):
    def __init__(self, T=5, d_model=32, nhead=4, num_layers=1):
        super().__init__()
        self.T = T
        self.input_dim = 7   # 6 treats + 1 vision per timestep
        self.output_dim = 6  # 6 belief logits

        self.embed = nn.Linear(self.input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, self.output_dim)

        # proper float mask for causal attention
        mask = torch.triu(torch.ones(T, T), diagonal=1)
        self.register_buffer("src_mask", mask.masked_fill(mask == 1, float("-inf")))

    def forward(self, treats, vision):
        B = treats.size(0)
        # combine treats and vision per timestep
        x = torch.cat([treats, vision.unsqueeze(-1)], dim=-1)  # [B, T, 7]
        x = self.embed(x.float())
        x = self.encoder(x, mask=self.src_mask[:self.T, :self.T])
        out = self.head(x)             # [B, T, 6]
        out = out[:, -1, :]            # last timestep
        return F.softmax(out, dim=-1)  # [B, 6]

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
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, uncertainty=0.0, archx='mlp'):
        self.archx = archx
        super().__init__(use_neural, random_prob, sigmoid_temp)
        self.uncertainty = uncertainty
        self.register_buffer('time_weights', torch.exp(torch.arange(5) * 2).view(1, 5))

    def _create_neural_network(self) -> nn.Module:
        if self.archx == "transformer":
            print('transformer beliefs!')
            return NormalizedBeliefTransformer()
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

        return beliefs 

class NormalizedTSBeliefNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(14)
        self.fc1 = nn.Linear(14, 12)
        self.fc2 = nn.Linear(12, 6)
        #self.fc3 = nn.Linear(16, 6)

    def forward(self, treats, vision, prev_beliefs, presence):
        batch_size = treats.shape[0]

        x = torch.cat([treats.reshape(batch_size, -1), vision.reshape(batch_size, -1), F.softmax(prev_beliefs, dim=-1).reshape(batch_size, -1), presence.reshape(batch_size, -1),], dim=1)
        x = self.input_norm(x)
        x = F.leaky_relu_(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, 6)
        #x = F.softmax(x, dim=-1)
        #x = torch.sigmoid(x)
        return x

class BeliefModulePerTimestep(BaseModule):
    def __init__(self, use_neural: bool = True, random_prob: float = 0.0, sigmoid_temp: float = 20.0, uncertainty=0.0):
        super().__init__(use_neural, random_prob, sigmoid_temp)
        self.uncertainty = uncertainty
        self.T = 5
        self.register_buffer('transition_counts_tensor', torch.zeros(5, 2, 2, 6, 6, 6, dtype=torch.int64), persistent=False)

    def _bin_beliefs(self, beliefs: torch.Tensor) -> list:
        quantized = torch.round(beliefs).long()
        return [tuple(row.tolist()) for row in quantized]

    def _create_neural_network(self) -> nn.Module:
        return NormalizedTSBeliefNetwork()

    @torch.no_grad()
    def _count_transitions(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, new_beliefs: torch.Tensor, t: int, presence: torch.Tensor, size_idx: int):
        if self.training:
            return
        B = visible_treats.size(0)
        prev_idx  = prev_beliefs.argmax(-1).long().view(B)
        cur_idx   = new_beliefs.argmax(-1).long().view(B)
        treat_idx = visible_treats.argmax(-1).long().view(B)
        v = vision.round().long().view(B)
        p = presence.view(B, -1).round().long().view(B)
        if p.max() == 0:
            return
        idx = ((prev_idx * 6 + treat_idx) * 6 + cur_idx)
        for vis in (0, 1):
            m = (v == vis) & (p == 1)
            if m.any():
                bins = torch.bincount(idx[m], minlength=216).view(6, 6, 6)
                self.transition_counts_tensor[t, size_idx, vis].add_(bins)
            
    def _hardcoded_forward(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, t: int, presence: torch.Tensor) -> torch.Tensor:
        time_weight = torch.exp(torch.tensor(t, dtype=torch.float32, device=visible_treats.device) * 2.0)
        update_mask = vision.unsqueeze(-1) * presence
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
        return beliefs

    def forward(self, visible_treats: torch.Tensor, vision: torch.Tensor, prev_beliefs: torch.Tensor, t: int, presence: torch.Tensor, size_idx: int=0) -> torch.Tensor:
        if self.use_neural:
            new_beliefs = self._neural_forward(visible_treats, vision, prev_beliefs, presence)
        else:
            self.sigmoid_temp = 90.0
            new_beliefs = self._hardcoded_forward(visible_treats, vision, prev_beliefs, t, presence)
        
        self._count_transitions(visible_treats, vision, prev_beliefs, new_beliefs, t, presence, size_idx)
        
        return new_beliefs

    def save_transition_table(self, filepath='transition_table.csv'):
        import pandas as pd
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tc = self.transition_counts_tensor.detach().cpu()
        nz = (tc > 0).nonzero(as_tuple=False)
        rows = []
        eye = torch.eye(6, dtype=torch.long)
        for t, s, v, i, k, j in nz:
            rows.append({
                'timestep': int(t.item()),
                'treat_size': int(s.item()),
                'from_state': str(tuple(eye[i].tolist())),
                'treat_state': str(tuple(eye[k].tolist())),
                'vision': int(v.item()),
                'to_state': str(tuple(eye[j].tolist())),
                'count': float(tc[t, s, v, i, k, j].item()),
            })
        df = pd.DataFrame(rows, columns=['timestep','treat_size','from_state','treat_state','vision','to_state','count'])
        df.to_csv(filepath, index=False)
        if len(df) == 0:
            return df, pd.DataFrame()
        pivot = df.pivot_table(index=['timestep','treat_size','from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
        pivot.to_csv(filepath.replace('.csv', '_pivot.csv'))
        for t in range(self.T):
            dft = df[df['timestep'] == t]
            if len(dft) > 0:
                dft.to_csv(filepath.replace('.csv', f'_t{t}.csv'), index=False)
                pivt = dft.pivot_table(index=['treat_size','from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
                pivt.to_csv(filepath.replace('.csv', f'_t{t}_pivot.csv'))
                for s in [0,1]:
                    dfts = dft[dft['treat_size'] == s]
                    if len(dfts) > 0:
                        dfts.to_csv(filepath.replace('.csv', f'_t{t}_s{s}.csv'), index=False)
                        pivts = dfts.pivot_table(index=['from_state','treat_state','vision'], columns='to_state', values='count', fill_value=0)
                        pivts.to_csv(filepath.replace('.csv', f'_t{t}_s{s}_pivot.csv'))
        return df, pivot




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
            nn.Linear(16, 6),
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

        large_choice = belief_vector[:, 0, :5]
        small_choice = belief_vector[:, 1, :5]

        both_no_treat = large_no_treat * small_no_treat
        large_exists = 1 - large_no_treat
        small_exists = 1 - small_no_treat

        greedy_decision = both_no_treat.unsqueeze(1) * default_choice + large_exists.unsqueeze(1) * large_choice + (1 - large_exists).unsqueeze(1) * small_exists.unsqueeze(1) * small_choice

        conflict = torch.max(large_choice * dominant_decision[:,:5], dim=1, keepdim=True)[0] # either sum or max works here... but sum needs more sigmoids
        #conflict = torch.sigmoid((conflict - 0.5))
        subordinate_decision = (1 - conflict) * large_choice + conflict * small_choice

        return dominant_present * subordinate_decision + (1 - dominant_present) * greedy_decision

    def _random_forward(self, belief_vector: torch.Tensor, dominant_decision: torch.Tensor = None, dominant_present: torch.Tensor = None) -> torch.Tensor:
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
        self.skip_sim_loss = not module_configs['use_sim_loss']
        
        self.process_opponent_perception = module_configs.get('opponent_perception', False)
        self.output_type = module_configs.get('output_type', 'my_decision')
        self.pad = module_configs.get('pad', '')
        self.size_swap = module_configs.get('size_swap', False)
        self.use_oracle = module_configs.get('use_oracle', False)
        print('size_swap', self.size_swap)
        self.use_per_timestep_opponent = False
        self.store_per_timestep_beliefs = False
        self.record_og_beliefs = False
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 2, 6))
        self.arch = module_configs['arch']

        self.sim_style = ""

        sigmoid_temp = module_configs.get('sigmoid_temp', 20.0)

        self.register_buffer('null_decision', torch.zeros(self.kwargs['batch_size'], 5))
        self.register_buffer('null_presence', torch.zeros(self.kwargs['batch_size'], 1))

        print('module_configs', module_configs)
        print('random_probs', random_probs)

        print("Ablation Architecture")

        if random_probs is None:
            random_probs = {k: 0.0 for k in module_configs.keys()}

        if module_configs.get('end2end'):
            self.end2end_model = EndToEndModel(arch=module_configs['arch'], output_type=self.output_type, pad=self.pad)
        else:
            print("not e2e")
            self.end2end_model = None

        self.treat_perception_my = TreatPerceptionModule(
            use_neural=module_configs['my_treat'],
            random_prob=random_probs['my_treat'],
            sigmoid_temp=sigmoid_temp, archx=module_configs['archx']
        )

        self.treat_perception_op = (TreatPerceptionModule(
            use_neural=module_configs['op_treat'],
            random_prob=random_probs['op_treat'],
            sigmoid_temp=sigmoid_temp, archx=module_configs['archx']
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
            sigmoid_temp=sigmoid_temp, uncertainty=0.0 if not module_configs['shared_belief'] else 0.0,
            archx=module_configs['archx']
        )

        self.op_belief = (BeliefModule(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0, 
            archx=module_configs['archx'],
        ) if not module_configs['shared_belief'] else self.my_belief) if not self.use_per_timestep_opponent else (BeliefModulePerTimestep(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0
        ) if not module_configs['shared_belief'] else self.my_belief)

        self.og_op_belief = BeliefModule(
            use_neural=module_configs['op_belief'],
            random_prob=random_probs['op_belief'],
            sigmoid_temp=sigmoid_temp, uncertainty=0.0, archx=module_configs['archx']
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
            'end2end_model': self.end2end_model
        }
    
    def get_trainable_modules(self):
        trainable = []
        for name, module in self.get_module_dict().items():
            if any(p.requires_grad for p in module.parameters()):
                trainable.append((name, module))
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

    def build_timestep_input(self, treats, vision, presence):
        timestep_chunks = []
        for t in range(5):
            chunk = torch.cat([
                treats[:, t].flatten(start_dim=1), 
                vision[:, t].unsqueeze(1),                  
                presence[:, t].unsqueeze(1)          
            ], dim=1)
            timestep_chunks.append(chunk)
        return torch.cat(timestep_chunks, dim=1)


    def forward(self, perceptual_field: torch.Tensor, additional_input: torch.Tensor) -> torch.Tensor:
        device = perceptual_field.device
        batch_size = perceptual_field.shape[0]

        printing = additional_input

        opponent_vision = self.vision_perception_op(perceptual_field, is_p1=0)
        opponent_presence = self.presence_perception_op(perceptual_field, is_p1=0)


        treats_op = self.treat_perception_op(perceptual_field)
        print(treats_op.shape)
        if len(treats_op.shape) == 2:
            treats_op = treats_op.reshape([-1, 5, 2, 6])
        treats_l_op, treats_s_op = treats_op[:, :, 0:2].unbind(2)
        
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
                end2end_input = self.build_timestep_input(treats_op, opponent_vision, opponent_presence_seq)
            else:
                end2end_input = perceptual_field
            return {
                'my_decision': self.end2end_model(end2end_input)
                }
        elif self.end2end_model is not None and self.output_type == 'multi':
            if self.process_opponent_perception:
                if True:
                    opponent_presence_seq = opponent_presence if opponent_presence.shape[1] != 1 else opponent_presence.repeat(1, 5)
                    #print(treats_op.shape)

                    end2end_input = self.build_timestep_input(treats_op, opponent_vision, opponent_presence_seq)

                    outputs_orig  = self.end2end_model(end2end_input)        # dict with [B,2,T,6]
                    if not self.size_swap:
                        op_belief_t = outputs_orig['op_belief_t']
                        my_belief_t = outputs_orig['my_belief_t']
                    else:
                        treats_swap = treats_op.flip(dims=[2])
                        end2end_input_swap = self.build_timestep_input(treats_swap, opponent_vision, opponent_presence_seq)
                        outputs_swap  = self.end2end_model(end2end_input_swap)   # dict with [B,2,T,6]
                        op_belief_t = 0.5 * (outputs_orig['op_belief_t'] + outputs_swap['op_belief_t'].flip(dims=[1]))
                        my_belief_t = 0.5 * (outputs_orig['my_belief_t'] + outputs_swap['my_belief_t'].flip(dims=[1]))

                    result = {
                        'op_belief_t': op_belief_t,                  # [B,2,T,6]
                        'my_belief_t': my_belief_t,                  # [B,2,T,6]
                        'op_decision_t': outputs_orig['op_decision_t'],  # [B,T,6]
                        'my_decision': outputs_orig['my_decision'],
                        'op_presence': opponent_presence,
                    }
                    return result

                else:
                    opponent_presence_seq = opponent_presence if opponent_presence.shape[1] != 1 else opponent_presence.repeat(1, 5)
                    gate = (opponent_vision * opponent_presence_seq).unsqueeze(-1).unsqueeze(-1)  # [B,5,1,1]
                    treats_op_gated = treats_op * gate + (1 - gate) * self.mask_token           # [B,5,2,6]

                    end2end_input_op = torch.cat([
                        treats_op_gated.flatten(start_dim=1),          # (B, 2*5*6)
                        #opponent_vision.flatten(start_dim=1),          # (B, 5)
                        #opponent_presence_seq.flatten(start_dim=1)     # (B, 5)
                    ], dim=1)

                    op_out = self.end2end_model(end2end_input_op)   # shared weights

                    op_belief_t   = op_out['op_belief_t']
                    op_decision_t = op_out['op_decision_t']

                    treats_my = treats_op              # [B,5,2,6]
                    my_presence_seq = torch.ones_like(opponent_presence_seq)
                    vis_full = torch.ones_like(opponent_vision)                         # [B,5]

                    end2end_input_my = torch.cat([
                        treats_my.flatten(start_dim=1),                                  # (B, 2*5*6)
                        #vis_full.flatten(start_dim=1),                                   # (B, 5)
                        #my_presence_seq.flatten(start_dim=1)                             # (B, 5)
                    ], dim=1)

                    my_out = self.end2end_model(end2end_input_my)     # same weights as above

                    my_belief_t  = my_out['my_belief_t']
                    my_decision  = my_out['my_decision']

                    return {
                        'op_belief_t':   op_belief_t,
                        'my_belief_t':   my_belief_t,
                        'op_decision_t': op_decision_t,
                        'my_decision':   my_decision,
                    }

            else:
                end2end_input = perceptual_field


            return self.end2end_model(end2end_input)  

        elif self.end2end_model is not None and self.output_type == 'op_belief':
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
                    op_beliefs_l_timesteps.append(op_belief_l)
                    op_beliefs_s_timesteps.append(op_belief_s)
                
                for t in range(1, 5):
                    #op_belief_l = self.op_belief.forward(treats_l_op[:, :t+1], opponent_vision[:, :t+1])
                    #op_belief_s = self.op_belief.forward(treats_s_op[:, :t+1], opponent_vision[:, :t+1])
                    op_belief_l = self.op_belief.forward(treats_l_op[:, t], opponent_vision[:, t], op_belief_l, t, opponent_presence, 0)
                    op_belief_s = self.op_belief.forward(treats_s_op[:, t], opponent_vision[:, t], op_belief_s, t, opponent_presence, 1)

                    if self.store_per_timestep_beliefs:
                        op_beliefs_l_timesteps.append(F.softmax(op_belief_l, dim=-1))
                        op_beliefs_s_timesteps.append(F.softmax(op_belief_s, dim=-1))

                #op_belief_l = 2.0*F.softmax(op_belief_l, dim=-1)
                #op_belief_s = 2.0*F.softmax(op_belief_s, dim=-1)

                if self.store_per_timestep_beliefs:
                    stacked_op_beliefs = torch.stack([
                        torch.stack(op_beliefs_l_timesteps, dim=1), 
                        torch.stack(op_beliefs_s_timesteps, dim=1)
                    ], dim=1)
                    if self.record_og_beliefs:
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
            op_belief_vector = op_beliefs
            op_decision = self.op_decision(op_belief_vector, self.null_decision[:batch_size].to(device), self.null_presence[:batch_size].to(device))

            if self.detach_decision:
                op_decision = op_decision.detach()

        masked_visions = (torch.rand(batch_size, self.num_visions, 5, device=device) <= self.vision_prob).float()

        self.my_belief.uncertainty = 0.0
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
        my_belief_vector = beliefs_tensor.squeeze(1)


        my_decision = self.my_decision.forward(my_belief_vector, op_decision, opponent_presence)

        if printing:
            v = (opponent_vision[printing] > 0.5)
            print('vision_op', [int(x) for x in range(len(v))], [float(x) for x in opponent_vision[printing].tolist()])
            print('Treats_my L:', treats_my[printing,:,0,:].argmax(1).int().tolist(),
                  'S:', treats_my[printing,:,1,:].argmax(1).int().tolist())
            print('Treats_op L:', torch.where(v, treats_op[printing,:,0,:].argmax(1), torch.tensor(5, device=v.device)).int().tolist(),
                  'S:', torch.where(v, treats_op[printing,:,1,:].argmax(1), torch.tensor(5, device=v.device)).int().tolist())
            print('my belief (L,S):', *(my_belief_vector[printing].argmax(1).int().tolist()))
            print('op belief (L,S):', *(op_belief_vector[printing].argmax(1).int().tolist()))
            print('op decision:', int(op_decision[printing].argmax()))
            print('my decision:', int(my_decision[printing].argmax()))

        #print(self.op_belief.summary())

        result = {
            'treat_perception': treats_op,
            'vision_perception': opponent_vision,
            'vision_perception_my': my_vision,
            'presence_perception': opponent_presence,
            'my_belief': my_belief_vector,
            'og_op_belief': og_op_beliefs if self.record_og_beliefs else None,
            'op_belief_t': stacked_op_beliefs if self.store_per_timestep_beliefs else None,
            'op_belief': op_belief_vector,
            'my_decision': my_decision,
            'op_decision': op_decision,
        }


        return result



class OpponentRawSimulator(nn.Module):
    def __init__(self, d_model=32, T=5, C=5, H=7, W=7, style="r"):
        super().__init__()
        self.T, self.C, self.H, self.W = T, C, H, W
        self.style = style
        in_dim  = C * H * W
        out_dim = C * H * W
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 4, 3, padding=1), nn.ReLU(),
            #nn.Conv2d(8, 4, 3, padding=1), nn.ReLU(),
        )
        self.head_r = nn.Linear(4*H*W, C*H*W)
        self.head_i = nn.Linear(4*H*W, C*H*W)
        self.head_p = nn.Linear(4*H*W, 1)

    def forward(self, x_enc):
        B, T, C, H, W = x_enc.shape
        x = x_enc.view(B * T, C, H, W) 
        x = self.encoder(x)
        x = x.view(B, T, 4 * H * W)
        out_r = self.head_r(x).view(B, T, self.C, self.H, self.W)
        out_i = self.head_i(x).view(B, T, self.C, self.H, self.W)
        out_p = self.head_p(x).view(B, T, 1, 1, 1).expand(B, T, self.C, self.H, self.W)

        if self.style == "r":
            return {"r": out_r}
        elif self.style == "i":
            return {"i": out_i}
        elif self.style == "rp":
            return {"r": out_r, "p": out_p}
        else:
            return {"r": out_r, "i": out_i}

    @torch.no_grad()
    def build_ground_truth(self, perceptual_field, vision, presence, style):
        B, T, C, H, W = perceptual_field.shape
        gt = perceptual_field.clone()

        gt_r, gt_i, gt_p = None, None, None
        if style in ("r", "ri", "rp"):
            eff_v_r = (vision * presence.view(B, 1)).view(B, T, 1, 1, 1)
            gt_r = gt.clone()
            gt_r[:, :, 1:2].mul_(eff_v_r)
        if style in ("i", "ri"):
            eff_v_i = vision.view(B, T, 1, 1, 1)
            gt_i = gt.clone()
            gt_i[:, :, 1:2].mul_(eff_v_i)
        if style in ("rp",):
            gt_p = presence.view(B, 1, 1, 1, 1).expand(B, T, C, H, W)

        if style == "r":
            return {"r": gt_r}
        elif style == "i":
            return {"i": gt_i}
        elif style == "rp":
            return {"r": gt_r, "p": gt_p}
        else: 
            return {"r": gt_r, "i": gt_i}

    def loss(self, pred, gt):
        return {k: F.mse_loss(pred[k], gt[k]) for k in pred if gt[k] is not None}

def print_perception(perception, batch_idx=0, time_idx=0):
    T, C, H, W = perception.shape[1], perception.shape[2], perception.shape[3], perception.shape[4]
    data = perception[batch_idx].detach().cpu().numpy()
    
    for t in range(T):
        for h in range(H):
            row_strs = []
            for c in range(C):
                row_strs.append(" ".join(f"{data[t, c, h, w]:1.0f}" for w in range(W)))
            print("   ".join(row_strs))
        print()

class SimulationEndToEnd(nn.Module):
    def __init__(self, module_configs: dict, random_probs, batch_size=256):
        super().__init__()
        self.module_configs = module_configs
        self.arch = module_configs["arch"]
        self.pad = module_configs.get("pad", "")
        self.output_type = module_configs.get("output_type", "multi")
        self.sim_style = module_configs.get("sim_type", "ri")[1:]
        self.kwargs = {'module_configs': module_configs}
        self.mode = "shared" if module_configs["shared"] else "split" if module_configs["split"] else "single"
        self.use_oracle = module_configs.get('use_oracle', False)
        self.skip_sim_loss = not module_configs['use_sim_loss']
        self.register_buffer("zero", torch.tensor(0.0))

        self.use_gt_sim = module_configs.get("use_gt_sim", False)

        print(module_configs)

        self.use_per_timestep_opponent = False
        self.store_per_timestep_beliefs = True
        self.record_og_beliefs = False

        print(self.mode)

        d_model = 32 if "32" in self.arch else 128

        print('padddd', self.pad, 'sim_loss skip', self.skip_sim_loss)

        in_channels = 3
        if self.mode == "single":
            in_channels += 3
        if self.sim_style in ("ri", "rp"):
            in_channels += 3

        self.in_channels = in_channels

        self.end2end_self = EndToEndModel(arch=self.arch, output_type=self.output_type, pad=self.pad, in_channels=in_channels if self.mode == "single" else 3 if self.sim_style not in ("ri", "rp") else 6)
        if self.mode == "shared":
            self.end2end_op = self.end2end_self
        elif self.mode == "split":
            self.end2end_op = EndToEndModel(arch=self.arch, output_type=self.output_type, pad=self.pad, in_channels=3 if self.sim_style not in ("ri", "rp") else 10)
        elif self.mode == "single":
            self.end2end_op = None

        self.simulator = OpponentRawSimulator(d_model=d_model, style=self.sim_style)

        sig = module_configs.get("sigmoid_temp", 20.0)
        print('sigmoid temp', sig)
        ax = module_configs.get("archx", "mlp")
        self.treat_op = TreatPerceptionModule(use_neural=False, random_prob=0.0, sigmoid_temp=sig, archx=ax)
        self.vision_op = VisionPerceptionModule(use_neural=False, random_prob=0.0, sigmoid_temp=sig)
        self.presence_op = PresencePerceptionModule(use_neural=False, random_prob=0.0, sigmoid_temp=sig)

        comb_in = 6
        if self.mode != "single":
            comb_in += 6

        self.combiner = nn.Sequential(nn.Linear(comb_in, d_model), nn.ReLU(), nn.Linear(d_model, 6))

    def forward(self, my_raw: torch.Tensor, additional_input: torch.Tensor):
        opp_vision = self.vision_op(my_raw, is_p1=0)
        opp_presence = self.presence_op(my_raw, is_p1=0)

        gt_raw = self.simulator.build_ground_truth(my_raw, opp_vision, opp_presence, self.sim_style)

        sim_pred = self.simulator(my_raw) if not self.use_gt_sim else gt_raw
        if self.sim_style == "r":
            sim_field = sim_pred["r"]
        elif self.sim_style == "i":
            sim_field = sim_pred["i"]
        elif self.sim_style == "ri":
            sim_field = torch.cat([sim_pred["r"], sim_pred["i"]], dim=2)
        elif self.sim_style == "rp":
            sim_field = torch.cat([sim_pred["r"], sim_pred["p"]], dim=2)

        #print("sim style", self.sim_style)
        #print("GT R")
        #print_perception(gt_raw["r"])
        #print("Raw")
        #print_perception(my_raw)

        if self.sim_style != "":

            if self.mode == "single":
                joint_raw = torch.cat([my_raw, sim_field], dim=2)
                joint_out = self.end2end_self(joint_raw)
                my_decision = joint_out["my_decision"]
            else:
                my_out = self.end2end_self(my_raw) if self.sim_style not in ("ri", "rp") else self.end2end_self(torch.cat([my_raw, my_raw], dim=2))
                op_out = self.end2end_op(sim_field)
                z_self = my_out["my_decision"]
                z_op = op_out["my_decision"]
                my_decision = self.combiner(torch.cat([z_self, z_op], dim=-1))

        else:
            if self.mode == "single":
                joint_out = self.end2end_self(torch.cat([my_raw, my_raw], dim=2))
                my_decision = joint_out["my_decision"]
            else:
                my_out = self.end2end_self(my_raw)
                op_out = self.end2end_op(my_raw)
                z_self = my_out["my_decision"]
                z_op = op_out["my_decision"]
                my_decision = self.combiner(torch.cat([z_self, z_op], dim=-1))

        return {
            "my_decision": my_decision,
            "sim_loss": {"r": self.zero, "i": self.zero, **self.simulator.loss(sim_pred, gt_raw)},
        }