'''
* @name: AGFmodule.py
* @description: Implementation of AGF
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch import Tensor
from torch.autograd import Variable

class AGFmodule(nn.Module):
    def __init__(self, n_heads, x_t, x_a, output, seq_len):
        super(AGFmodule, self).__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.output = output
        
        self.q_linear = nn.Linear(x_t, output * n_heads)
        self.k_linear = nn.Linear(x_a, output * n_heads)
        
        self.v_linear = nn.Parameter(torch.randn(seq_len, output * n_heads))
        
        self.out = nn.Linear(output * n_heads, output)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x_text,x_a):
        device = x_text.device
        self.v_linear = self.v_linear.to(device)
        
        batch_size = x_text.size(0)
        
        q = self.q_linear(x_text)
        k = self.k_linear(x_global)

        q = q.view(batch_size, self.seq_len, self.n_heads, -1).transpose(1, 2)
        k = k.view(batch_size, self.seq_len, self.n_heads, -1).transpose(1, 2)
        
        v = self.v_linear.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.seq_len, self.n_heads, -1).transpose(1, 2)
        v = v.contiguous()

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, self.seq_len, -1)
        
        output = self.out(context)
        
        return output
