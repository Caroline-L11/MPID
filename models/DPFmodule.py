'''
* @name: DPFmodule.py
* @description: Implementation of DPF
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch import Tensor
from torch.autograd import Variable

class DistanceFusionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1_v = nn.Linear(dim, self.mlp_hidden_dim)
        self.fc1_a = nn.Linear(dim, self.mlp_hidden_dim)

        self.mlp_v = nn.Linear(self.mlp_hidden_dim, dim)
        self.mlp_a = nn.Linear(self.mlp_hidden_dim, dim)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc_out = nn.Linear(dim * 2, dim)

    def forward(self, x_v, x_a):
        manhattan_dist = torch.sum(torch.abs(x_v.unsqueeze(2) - x_a.unsqueeze(1)), dim=-1)  # [batch_size, seq_len_v, seq_len_a]

        manhattan_dist_v = manhattan_dist.mean(dim=2)  # [batch_size, seq_len_v]
        manhattan_dist_a = manhattan_dist.mean(dim=1)  # [batch_size, seq_len_a]
        
        x_v_weighted = x_v * manhattan_dist_v.unsqueeze(-1)  # [batch_size, seq_len_v, dim]
        x_a_weighted = x_a * manhattan_dist_a.unsqueeze(-1)  # [batch_size, seq_len_a, dim]
        
        x_v = self.act(self.fc1_v(x_v_weighted))
        x_v = self.drop(self.mlp_v(x_v))
        
        x_a = self.act(self.fc1_a(x_a_weighted))
        x_a = self.drop(self.mlp_a(x_a))
        
        combined_features = torch.cat([x_v, x_a], dim=-1)
        
        output = self.fc_out(combined_features)
        
        return output

class SemanticDistanceFusion(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 gate_bias=False):
        super().__init__()
        self.semantic_block = SemanticFusionBlock(
            dim, num_heads, attn_ratio, mlp_ratio, qkv_bias,
            drop, attn_drop, drop_path, act_layer, norm_layer)
        self.distance_block = DistanceFusionBlock(dim, mlp_ratio, drop, act_layer)
        
        self.fusion_fc = nn.Linear(dim * 2, 2)
        self.semantic_gate_act = torch.sigmoid
        self.distance_gate_act = torch.tanh

    def forward(self, xmm, xv, xa):
        # Process semantic output
        semantic_output = self.semantic_block(xmm, xv, xa)
        
        # Process distance output
        distance_output = self.distance_block(xv, xa)  # Using xv and xa for distance fusion

        # Ensure the dimensions match for gating
        batch_size, seq_len, _ = semantic_output.size()
        distance_output = distance_output[:, :seq_len, :]  # Match sequence length if needed

        # Concatenate the outputs in the feature dimension
        combined_output = torch.cat([semantic_output, distance_output], dim=-1)
        
        # Compute gate coefficients for the combined output
        gate_coeffs = self.fusion_fc(combined_output)
        
        # Apply activations and ensure correct shapes for gating
        semantic_gate_coeff = self.semantic_gate_act(gate_coeffs[:, :, 0]).unsqueeze(-1)  # Shape (batch_size, seq_len, 1)
        distance_gate_coeff = self.distance_gate_act(gate_coeffs[:, :, 1]).unsqueeze(-1)  # Shape (batch_size, seq_len, 1)
        
        # Apply gate coefficients
        gated_semantic_output = semantic_gate_coeff * semantic_output
        gated_distance_output = distance_gate_coeff * distance_output
        
        # Combine the gated outputs
        fusion_output = gated_semantic_output + gated_distance_output

        return fusion_output
