#! this .py file heavily borrows from https://github.com/yujiangpu20/PEL4VAD/tree/master, thanks for their great codes
import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn.parameter import Parameter
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy as dcp
import pdb


# ============================================================================
# Complex-Valued Self-Attention Components
# ============================================================================

class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.
    For complex input z = x + iy and complex weight W = A + iB:
    W @ z = (Ax - By) + i(Ay + Bx)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features, bias=bias)
        self.linear_imag = nn.Linear(in_features, out_features, bias=bias)

        # NUMERICAL STABILITY: Initialize with smaller weights to prevent explosion
        # Xavier/Glorot initialization scaled down for complex arithmetic
        nn.init.xavier_uniform_(self.linear_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.linear_imag.weight, gain=0.5)
        if bias:
            nn.init.zeros_(self.linear_real.bias)
            nn.init.zeros_(self.linear_imag.bias)

    def forward(self, real, imag):
        # (A + iB)(x + iy) = (Ax - By) + i(Ay + Bx)
        out_real = self.linear_real(real) - self.linear_imag(imag)
        out_imag = self.linear_real(imag) + self.linear_imag(real)
        return out_real, out_imag


class ComplexMultiHeadAttention(nn.Module):
    """
    Complex-valued Multi-Head Attention (Proper Implementation).

    Based on: "Building Blocks for a Complex-Valued Transformer Architecture"

    Key differences from naive implementation:
    1. Compute BOTH real and imaginary parts of attention scores
    2. Apply softmax to MAGNITUDE (to get valid probability distribution)
    3. PRESERVE phase information in attention weights
    4. Apply COMPLEX multiplication: (attn_r + i*attn_i) @ (V_r + i*V_i)

    This preserves phase information through the attention mechanism,
    which is critical for phase-coherence based learning.
    """
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm_flag=None):
        super(ComplexMultiHeadAttention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.norm_flag = norm_flag

        # Complex Q, K, V projections
        self.q = ComplexLinear(d_model, dim_k)
        self.k = ComplexLinear(d_model, dim_k)
        self.v = ComplexLinear(d_model, dim_v)
        self.o = ComplexLinear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)

    def forward(self, query_real, query_imag, key_real, key_imag, value_real, value_imag, mask=None, adj=None):
        """
        Args:
            query_real, query_imag: [bs, 10, d_model] - complex query
            key_real, key_imag: [bs, 10, d_model] - complex key
            value_real, value_imag: [bs, 10, d_model] - complex value
        Returns:
            out_real, out_imag: [bs, 10, d_model] - complex output
        """
        bs = query_real.shape[0]
        seq_len = query_real.shape[1]

        # Complex Q, K, V projections
        Q_real, Q_imag = self.q(query_real, query_imag)
        K_real, K_imag = self.k(key_real, key_imag)
        V_real, V_imag = self.v(value_real, value_imag)

        # Reshape for multi-head: [bs, seq, dim] -> [bs, heads, seq, dim/heads]
        Q_real = Q_real.view(bs, seq_len, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        Q_imag = Q_imag.view(bs, seq_len, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        K_real = K_real.view(bs, seq_len, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        K_imag = K_imag.view(bs, seq_len, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2)
        V_real = V_real.view(bs, seq_len, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2)
        V_imag = V_imag.view(bs, seq_len, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2)

        # =====================================================================
        # PROPER COMPLEX ATTENTION (per "Building Blocks for Complex Transformer")
        # =====================================================================
        # Complex attention scores: Q @ K^H where K^H is conjugate transpose
        # (Q_r + i*Q_i) @ (K_r - i*K_i)^T = (Q_r·K_r^T + Q_i·K_i^T) + i*(Q_i·K_r^T - Q_r·K_i^T)

        # Real part of attention scores
        attn_real = torch.matmul(Q_real, K_real.transpose(-2, -1)) + torch.matmul(Q_imag, K_imag.transpose(-2, -1))

        # Imaginary part of attention scores (previously DISCARDED - now KEPT)
        attn_imag = torch.matmul(Q_imag, K_real.transpose(-2, -1)) - torch.matmul(Q_real, K_imag.transpose(-2, -1))

        # Scale both parts
        attn_real = attn_real * self.norm_fact  # [bs, heads, seq, seq]
        attn_imag = attn_imag * self.norm_fact  # [bs, heads, seq, seq]

        # Compute magnitude and phase of complex attention scores
        attn_magnitude = torch.sqrt(attn_real ** 2 + attn_imag ** 2 + 1e-8)  # [bs, heads, seq, seq]
        attn_phase = torch.atan2(attn_imag, attn_real + 1e-8)  # [bs, heads, seq, seq]

        if adj is not None:
            attn_magnitude = attn_magnitude + adj

        if mask is not None:
            attn_magnitude = attn_magnitude.masked_fill_(mask.data.eq(0), -1e9)

        # Softmax on MAGNITUDE (gives valid probability distribution)
        attn_weights_mag = self.act(attn_magnitude)  # [bs, heads, seq, seq] - real, sums to 1

        # Reconstruct complex attention weights: magnitude * e^(i*phase)
        attn_weights_real = attn_weights_mag * torch.cos(attn_phase)  # [bs, heads, seq, seq]
        attn_weights_imag = attn_weights_mag * torch.sin(attn_phase)  # [bs, heads, seq, seq]

        # =====================================================================
        # COMPLEX MULTIPLICATION: (attn_r + i*attn_i) @ (V_r + i*V_i)
        # = (attn_r @ V_r - attn_i @ V_i) + i*(attn_r @ V_i + attn_i @ V_r)
        # =====================================================================
        out_real = torch.matmul(attn_weights_real, V_real) - torch.matmul(attn_weights_imag, V_imag)
        out_imag = torch.matmul(attn_weights_real, V_imag) + torch.matmul(attn_weights_imag, V_real)

        # Reshape back: [bs, heads, seq, dim/heads] -> [bs, seq, dim]
        out_real = out_real.transpose(1, 2).reshape(bs, seq_len, -1)
        out_imag = out_imag.transpose(1, 2).reshape(bs, seq_len, -1)

        # Output projection
        out_real, out_imag = self.o(out_real, out_imag)

        return out_real, out_imag


class ComplexPositionwiseFeedForward(nn.Module):
    """Complex-valued FFN with component-wise ReLU on real/imag."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ComplexPositionwiseFeedForward, self).__init__()
        self.w_1 = ComplexLinear(d_model, d_ff)
        self.w_2 = ComplexLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, real, imag):
        real, imag = self.w_1(real, imag)
        real = F.relu(real)
        imag = F.relu(imag)
        real = self.dropout(real)
        imag = self.dropout(imag)
        real, imag = self.w_2(real, imag)
        return real, imag


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex values (applied separately)."""
    def __init__(self, d_model):
        super(ComplexLayerNorm, self).__init__()
        self.norm_real = nn.LayerNorm(d_model)
        self.norm_imag = nn.LayerNorm(d_model)

    def forward(self, real, imag):
        return self.norm_real(real), self.norm_imag(imag)


class ComplexTransformerEncoder(nn.Module):
    """
    Complex-valued Transformer Encoder.

    Processes audio (real) and visual (imag) jointly through complex operations.
    This naturally creates cross-modal interaction through complex arithmetic.
    """
    def __init__(
        self,
        d_model=1024,
        hid_dim=256,
        d_ff=512,
        n_heads=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(ComplexTransformerEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = ComplexMultiHeadAttention(d_model, hid_dim, hid_dim, n_heads, norm_flag)
        self.ffn = ComplexPositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = ComplexLayerNorm(d_model)
        self.norm2 = ComplexLayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

    def forward(self, real, imag):
        """
        Args:
            real: [bs, 10, 1024] - audio features
            imag: [bs, 10, 1024] - visual features
        Returns:
            real, imag: [bs, 10, 1024] - processed features
        """
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(real.shape[0], real.shape[1], self.n_heads)
        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, real.shape[1], real.shape[0])

        # Complex self-attention with residual
        attn_real, attn_imag = self.self_attn(real, imag, real, imag, real, imag, mask, adj)
        real = real + attn_real
        imag = imag + attn_imag
        real, imag = self.norm1(real, imag)

        # Complex FFN with residual
        ffn_real, ffn_imag = self.ffn(real, imag)
        real = real + ffn_real
        imag = imag + ffn_imag
        real, imag = self.norm2(real, imag)

        return real, imag

    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
        m = m.repeat(bs, self.n_heads, 1, 1).cuda()
        return m


# ============================================================================
# Original Real-Valued Components (kept for compatibility)
# ============================================================================

class DistanceAdj(nn.Module):
    def __init__(self, sigma=0.6, bias=0.2):
        super(DistanceAdj, self).__init__()
        # self.sigma = sigma
        # self.bias = bias
        #! not learnable
        # self.w = sigma
        # self.b = bias
        #! learnable
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len, n_heads):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).cuda()
        # dist = torch.from_numpy(squareform(dist))
        # dist = torch.exp(-self.sigma * dist ** 2) # no use
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_heads, 1, 1)

        return dist



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm_flag=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm_flag = norm_flag

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.FloatTensor([0]))


    def forward(self, query, key, value, mask=None, adj=None, return_attn_weights=False):
        # query, key, value: [bs, 10, 1024]
        bs = query.shape[0]
        Q = self.q(query).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2) # [bs, 10, h, dim_k/h] -> [bs, h, 10, dim_k/h]
        K = self.k(key).view(bs, -1, self.n_heads, self.dim_k // self.n_heads).transpose(1, 2) # -> [bs, h, 10, dim_k/h]
        V = self.v(value).view(bs, -1, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2) # -> [bs, h, 10, dim_v/h]

        if adj is not None:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_fact # [bs, h, 10, 10]

        if mask is not None:
            l_map = g_map.clone()
            l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)
        # pdb.set_trace()

        g_map = self.act(g_map) # [bs, h, 10, 10]
        attn_weights = g_map  # Store attention weights for distillation [bs, h, 10, 10]
        glb = torch.matmul(g_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1) # [bs, h, 10, dim_v/h] -> [bs, 10, h, dim_v/h] -> [bs, 10, dim_v]

        if mask is not None:
            l_map = self.act(l_map)
            lcl = torch.matmul(l_map, V).transpose(1, 2).reshape(bs, query.shape[1], -1) # -> [bs, 10, dim_v]
            alpha = torch.sigmoid(self.alpha)
            tmp = alpha * glb + (1 - alpha) * lcl
            # tmp = lcl
        else:
            tmp = glb
        # pdb.set_trace()

        if self.norm_flag:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm
        # tmp = self.o(tmp).view(-1, x.shape[1], x.shape[2]) # [bs, 10, d_model]
        tmp = self.o(tmp) # [bs, 10, d_model]
        # pdb.set_trace()

        if return_attn_weights:
            return tmp, attn_weights
        return tmp


class PositionwiseFeedForward(nn.Module):
    """Standard FFN: w_2( dropout( relu( w_1(x) ) ) )."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=1024,
        hid_dim=256,
        d_ff = 512,
        # out_dim,
        n_heads=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TransformerEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = MultiHeadAttention(d_model, hid_dim, hid_dim, n_heads, norm_flag)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.linear1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        # self.linear2 = nn.Conv1d(d_model // 2, out_dim, kernel_size=1)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

    def forward(self, q, k, v, return_attn_weights=False):
        # q, k, v: [bs, 10, 1024]
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(q.shape[0], q.shape[1], self.n_heads) # [bs, n_heads, 10, 10]
        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, q.shape[1], q.shape[0])
        # Multi-Head Attention, post-LayerNorm is used.
        if return_attn_weights:
            attn_out, attn_weights = self.self_attn(q, k, v, mask, adj, return_attn_weights=True)
            x = q + attn_out  # [bs, 10, d_model]
        else:
            x = q + self.self_attn(q, k, v, mask, adj) # [bs, 10, d_model]
            attn_weights = None
        x = self.norm1(x) # [bs, 10, d_model]
        # pdb.set_trace()

        # Feed-Forward Network, post-LayerNorm is used.
        x = x + self.ffn(x)
        x = self.norm2(x) # [bs, 10, d_model]
        # pdb.set_trace()

        # x = self.dropout1(F.gelu(self.linear1(x)))
        # x_e = self.dropout2(F.gelu(self.linear2(x)))
        if return_attn_weights:
            return x, attn_weights
        return x
        # return x_e, x


    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

        m = m.repeat(bs, self.n_heads, 1, 1).cuda() # [bs, n_head, 10, 10]
        # m = m.repeat(bs, self.n_heads, 1, 1) # [bs, n_head, 10, 10]
        return m


class TemporalAttentionModule(nn.Module):
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TemporalAttentionModule, self).__init__()
        block = TransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        #! parameter-shared in each layer and modality
        # self.a_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.a_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        # self.v_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.v_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        
        #! update 2024.10.26, correct version for parameter unshared
        self.a_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.a_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        self.v_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.v_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        # self.alpha_a = nn.Parameter(torch.FloatTensor([0.]))
        # self.alpha_v = nn.Parameter(torch.FloatTensor([0.]))


    def forward(self, a_tokens, v_tokens, return_attn_weights=False):
        # pdb.set_trace()

        # Collect attention weights from all layers if requested
        a_attn_weights_list = []
        v_attn_weights_list = []

        # ori_a, ori_v = a_tokens, v_tokens
        for blk_id, blk in enumerate(self.a_sa_blocks):
            if return_attn_weights:
                a_tokens, a_attn = blk(a_tokens, a_tokens, a_tokens, return_attn_weights=True)
                a_attn_weights_list.append(a_attn)
            else:
                a_tokens = blk(a_tokens, a_tokens, a_tokens)
        a_sa_tokens = a_tokens
        # pdb.set_trace()

        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.a_xa_blocks):
        #     a_tokens = blk(a_tokens, v_tokens, v_tokens)
        # a_xa_tokens = a_tokens
        # pdb.set_trace()


        for blk_id, blk in enumerate(self.v_sa_blocks):
            if return_attn_weights:
                v_tokens, v_attn = blk(v_tokens, v_tokens, v_tokens, return_attn_weights=True)
                v_attn_weights_list.append(v_attn)
            else:
                v_tokens = blk(v_tokens, v_tokens, v_tokens)
        v_sa_tokens = v_tokens
        # pdb.set_trace()

        # v_tokens = ori_v
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.v_xa_blocks):
        #     v_tokens = blk(v_tokens, a_tokens, a_tokens)
        # v_xa_tokens = v_tokens
        # pdb.set_trace()

        #! ablation: not learnable
        # a_tokens = a_sa_tokens + a_xa_tokens
        # v_tokens = v_sa_tokens + v_xa_tokens
        # a_tokens = (a_sa_tokens + a_xa_tokens) * 0.5
        # v_tokens = (v_sa_tokens + v_xa_tokens) * 0.5

        #! only self attention
        a_tokens = a_sa_tokens
        v_tokens = v_sa_tokens
        #! only cross attention
        # a_tokens = a_xa_tokens
        # v_tokens = v_xa_tokens


        #! learnable, turn to nan
        # alpha_a = torch.sigmoid(self.alpha_a)
        # a_tokens = alpha_a * a_sa_tokens + (1 - alpha_a) * a_xa_tokens
        # alpha_v = torch.sigmoid(self.alpha_v)
        # v_tokens = alpha_v * v_sa_tokens + (1 - alpha_v) * v_xa_tokens
        # # pdb.set_trace()

        if  torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        if return_attn_weights:
            # Stack attention weights from all layers: [num_layers, bs, heads, T, T]
            # For single layer, just return the weights directly: [bs, heads, T, T]
            a_attn_weights = torch.stack(a_attn_weights_list, dim=0) if len(a_attn_weights_list) > 1 else a_attn_weights_list[0] if a_attn_weights_list else None
            v_attn_weights = torch.stack(v_attn_weights_list, dim=0) if len(v_attn_weights_list) > 1 else v_attn_weights_list[0] if v_attn_weights_list else None
            return a_tokens, v_tokens, {'audio': a_attn_weights, 'visual': v_attn_weights}

        return a_tokens, v_tokens


class ComplexTemporalAttentionModule(nn.Module):
    """
    Complex-valued Temporal Attention Module.

    Treats audio as REAL part and visual as IMAGINARY part of complex numbers.
    The complex self-attention naturally creates cross-modal interaction through
    complex arithmetic operations.

    Key insight: In complex multiplication (A + iB) @ (C + iD)^H,
    the result mixes A with D and B with C, creating cross-modal interaction.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(ComplexTemporalAttentionModule, self).__init__()

        block = ComplexTransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Stack of complex transformer layers
        self.complex_blocks = nn.ModuleList([dcp(block) for i in range(sa_layer_num)])

    def forward(self, a_tokens, v_tokens):
        """
        Args:
            a_tokens: [bs, 10, 1024] - audio features (treated as REAL part)
            v_tokens: [bs, 10, 1024] - visual features (treated as IMAGINARY part)
        Returns:
            a_tokens: [bs, 10, 1024] - processed audio (from REAL part)
            v_tokens: [bs, 10, 1024] - processed visual (from IMAGINARY part)
        """
        # Audio = Real, Visual = Imaginary
        real, imag = a_tokens, v_tokens

        # Pass through complex transformer blocks
        for blk in self.complex_blocks:
            real, imag = blk(real, imag)

        # Output: Real -> Audio, Imag -> Visual
        a_tokens, v_tokens = real, imag

        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens


class SeparateComplexTemporalAttentionModule(nn.Module):
    """
    Separate Complex-Valued Temporal Attention Module.

    Each modality (audio and visual) is processed through its OWN complex transformer
    independently. The feature dimension is split in half to create real/imaginary
    components for each modality.

    Unlike ComplexTemporalAttentionModule which combines audio/visual as real/imag,
    this module:
        - Audio: features[:, :, :dim//2] = real, features[:, :, dim//2:] = imag
        - Visual: features[:, :, :dim//2] = real, features[:, :, dim//2:] = imag
        - Each processed by its own ComplexTransformerEncoder
        - No cross-modal mixing occurs

    This allows each modality to benefit from complex-valued operations (which can
    model phase and magnitude relationships) while keeping the streams separate.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(SeparateComplexTemporalAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.half_dim = feat_dim // 2

        # Create complex transformer encoder block for audio
        audio_block = ComplexTransformerEncoder(
            d_model=self.half_dim,  # Half dimension since we split into real/imag
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Create complex transformer encoder block for visual
        visual_block = ComplexTransformerEncoder(
            d_model=self.half_dim,  # Half dimension since we split into real/imag
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate stacks for each modality (no parameter sharing)
        self.audio_blocks = nn.ModuleList([dcp(audio_block) for _ in range(sa_layer_num)])
        self.visual_blocks = nn.ModuleList([dcp(visual_block) for _ in range(sa_layer_num)])

    def forward(self, a_tokens, v_tokens):
        """
        Args:
            a_tokens: [bs, 10, 1024] - audio features
            v_tokens: [bs, 10, 1024] - visual features
        Returns:
            a_tokens: [bs, 10, 1024] - processed audio features
            v_tokens: [bs, 10, 1024] - processed visual features
        """
        # Split audio into real and imaginary components
        a_real = a_tokens[:, :, :self.half_dim]  # [bs, 10, 512]
        a_imag = a_tokens[:, :, self.half_dim:]  # [bs, 10, 512]

        # Split visual into real and imaginary components
        v_real = v_tokens[:, :, :self.half_dim]  # [bs, 10, 512]
        v_imag = v_tokens[:, :, self.half_dim:]  # [bs, 10, 512]

        # Process audio through its own complex self-attention stream
        for blk in self.audio_blocks:
            a_real, a_imag = blk(a_real, a_imag)

        # Process visual through its own complex self-attention stream
        for blk in self.visual_blocks:
            v_real, v_imag = blk(v_real, v_imag)

        # Concatenate real and imaginary back together
        a_tokens = torch.cat([a_real, a_imag], dim=-1)  # [bs, 10, 1024]
        v_tokens = torch.cat([v_real, v_imag], dim=-1)  # [bs, 10, 1024]

        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens


class ComplexCrossModalAttentionModule(nn.Module):
    """
    Complex Cross-Modal Attention Module.

    Each modality first undergoes complex self-attention, then cross-attends to
    the other modality using complex-valued operations.

    Architecture:
        1. Split audio/visual into real/imag (half each)
        2. Complex self-attention: audio→audio, visual→visual
        3. Complex cross-attention: audio→visual, visual→audio
        4. Combine with residual connections

    This allows explicit cross-modal interaction while using complex operations.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Used for cross-attention layers
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(ComplexCrossModalAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.half_dim = feat_dim // 2
        self.n_heads = head_num
        self.win_size = win_size
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

        # Complex self-attention for audio
        self.audio_self_attn = ComplexMultiHeadAttention(
            d_model=self.half_dim,
            dim_k=hid_dim,
            dim_v=hid_dim,
            n_heads=head_num,
            norm_flag=norm_flag
        )
        self.audio_self_ffn = ComplexPositionwiseFeedForward(self.half_dim, d_ff, dropout)
        self.audio_self_norm1 = ComplexLayerNorm(self.half_dim)
        self.audio_self_norm2 = ComplexLayerNorm(self.half_dim)

        # Complex self-attention for visual
        self.visual_self_attn = ComplexMultiHeadAttention(
            d_model=self.half_dim,
            dim_k=hid_dim,
            dim_v=hid_dim,
            n_heads=head_num,
            norm_flag=norm_flag
        )
        self.visual_self_ffn = ComplexPositionwiseFeedForward(self.half_dim, d_ff, dropout)
        self.visual_self_norm1 = ComplexLayerNorm(self.half_dim)
        self.visual_self_norm2 = ComplexLayerNorm(self.half_dim)

        # Complex cross-attention: audio queries visual
        self.audio_cross_attn = ComplexMultiHeadAttention(
            d_model=self.half_dim,
            dim_k=hid_dim,
            dim_v=hid_dim,
            n_heads=head_num,
            norm_flag=norm_flag
        )
        self.audio_cross_ffn = ComplexPositionwiseFeedForward(self.half_dim, d_ff, dropout)
        self.audio_cross_norm1 = ComplexLayerNorm(self.half_dim)
        self.audio_cross_norm2 = ComplexLayerNorm(self.half_dim)

        # Complex cross-attention: visual queries audio
        self.visual_cross_attn = ComplexMultiHeadAttention(
            d_model=self.half_dim,
            dim_k=hid_dim,
            dim_v=hid_dim,
            n_heads=head_num,
            norm_flag=norm_flag
        )
        self.visual_cross_ffn = ComplexPositionwiseFeedForward(self.half_dim, d_ff, dropout)
        self.visual_cross_norm1 = ComplexLayerNorm(self.half_dim)
        self.visual_cross_norm2 = ComplexLayerNorm(self.half_dim)

        # Distance adjacency for temporal modeling
        self.loc_adj = DistanceAdj(gamma, bias)

    def forward(self, a_tokens, v_tokens):
        """
        Args:
            a_tokens: [bs, 10, 1024] - audio features
            v_tokens: [bs, 10, 1024] - visual features
        Returns:
            a_tokens: [bs, 10, 1024] - processed audio features
            v_tokens: [bs, 10, 1024] - processed visual features
        """
        bs = a_tokens.shape[0]
        seq_len = a_tokens.shape[1]

        # Split audio into real and imaginary components
        a_real = a_tokens[:, :, :self.half_dim]  # [bs, 10, 512]
        a_imag = a_tokens[:, :, self.half_dim:]  # [bs, 10, 512]

        # Split visual into real and imaginary components
        v_real = v_tokens[:, :, :self.half_dim]  # [bs, 10, 512]
        v_imag = v_tokens[:, :, self.half_dim:]  # [bs, 10, 512]

        # Get adjacency matrix if needed
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(bs, seq_len, self.n_heads)

        # ============================================
        # Step 1: Complex Self-Attention
        # ============================================

        # Audio self-attention
        a_sa_real, a_sa_imag = self.audio_self_attn(
            a_real, a_imag,  # Q
            a_real, a_imag,  # K
            a_real, a_imag,  # V
            adj=adj
        )
        a_real = a_real + a_sa_real
        a_imag = a_imag + a_sa_imag
        a_real, a_imag = self.audio_self_norm1(a_real, a_imag)

        a_ffn_real, a_ffn_imag = self.audio_self_ffn(a_real, a_imag)
        a_real = a_real + a_ffn_real
        a_imag = a_imag + a_ffn_imag
        a_real, a_imag = self.audio_self_norm2(a_real, a_imag)

        # Visual self-attention
        v_sa_real, v_sa_imag = self.visual_self_attn(
            v_real, v_imag,  # Q
            v_real, v_imag,  # K
            v_real, v_imag,  # V
            adj=adj
        )
        v_real = v_real + v_sa_real
        v_imag = v_imag + v_sa_imag
        v_real, v_imag = self.visual_self_norm1(v_real, v_imag)

        v_ffn_real, v_ffn_imag = self.visual_self_ffn(v_real, v_imag)
        v_real = v_real + v_ffn_real
        v_imag = v_imag + v_ffn_imag
        v_real, v_imag = self.visual_self_norm2(v_real, v_imag)

        # ============================================
        # Step 2: Complex Cross-Attention
        # ============================================

        # Audio cross-attends to Visual (Q=audio, K,V=visual)
        a_ca_real, a_ca_imag = self.audio_cross_attn(
            a_real, a_imag,  # Q from audio
            v_real, v_imag,  # K from visual
            v_real, v_imag,  # V from visual
            adj=adj
        )
        a_real = a_real + a_ca_real
        a_imag = a_imag + a_ca_imag
        a_real, a_imag = self.audio_cross_norm1(a_real, a_imag)

        a_ffn_real, a_ffn_imag = self.audio_cross_ffn(a_real, a_imag)
        a_real = a_real + a_ffn_real
        a_imag = a_imag + a_ffn_imag
        a_real, a_imag = self.audio_cross_norm2(a_real, a_imag)

        # Visual cross-attends to Audio (Q=visual, K,V=audio)
        v_ca_real, v_ca_imag = self.visual_cross_attn(
            v_real, v_imag,  # Q from visual
            a_real, a_imag,  # K from audio (after audio's cross-attn)
            a_real, a_imag,  # V from audio
            adj=adj
        )
        v_real = v_real + v_ca_real
        v_imag = v_imag + v_ca_imag
        v_real, v_imag = self.visual_cross_norm1(v_real, v_imag)

        v_ffn_real, v_ffn_imag = self.visual_cross_ffn(v_real, v_imag)
        v_real = v_real + v_ffn_real
        v_imag = v_imag + v_ffn_imag
        v_real, v_imag = self.visual_cross_norm2(v_real, v_imag)

        # ============================================
        # Step 3: Concatenate back
        # ============================================
        a_tokens = torch.cat([a_real, a_imag], dim=-1)  # [bs, 10, 1024]
        v_tokens = torch.cat([v_real, v_imag], dim=-1)  # [bs, 10, 1024]

        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens


class TemporalLinearModule(nn.Module):
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(TemporalLinearModule, self).__init__()
        block = nn.Linear(feat_dim, feat_dim)

        # self.a_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.a_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        # self.v_sa_blocks = nn.Sequential(*[block for i in range(sa_layer_num)])
        # self.v_xa_blocks = nn.Sequential(*[block for i in range(xa_layer_num)])
        
        #! update 2024.10.26, correct version for parameter unshared
        self.a_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.a_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        self.v_sa_blocks = nn.Sequential(*[dcp(block) for i in range(sa_layer_num)])
        self.v_xa_blocks = nn.Sequential(*[dcp(block) for i in range(xa_layer_num)])
        

        # self.alpha_a = nn.Parameter(torch.FloatTensor([0.]))
        # self.alpha_v = nn.Parameter(torch.FloatTensor([0.]))
        

    def forward(self, a_tokens, v_tokens):
        # pdb.set_trace()

        # ori_a, ori_v = a_tokens, v_tokens
        for blk_id, blk in enumerate(self.a_sa_blocks):
            a_tokens = blk(a_tokens)
        a_sa_tokens = a_tokens
        # pdb.set_trace()
        
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.a_xa_blocks):
        #     a_tokens = blk(a_tokens, v_tokens, v_tokens)
        # a_xa_tokens = a_tokens
        # pdb.set_trace()


        for blk_id, blk in enumerate(self.v_sa_blocks):
            v_tokens = blk(v_tokens)
        v_sa_tokens = v_tokens
        # pdb.set_trace()

        # v_tokens = ori_v
        # a_tokens = ori_a
        # for blk_id, blk in enumerate(self.v_xa_blocks):
        #     v_tokens = blk(v_tokens, a_tokens, a_tokens)
        # v_xa_tokens = v_tokens
        # pdb.set_trace()
        
        #! ablation: not learnable
        # a_tokens = a_sa_tokens + a_xa_tokens
        # v_tokens = v_sa_tokens + v_xa_tokens
        # a_tokens = (a_sa_tokens + a_xa_tokens) * 0.5
        # v_tokens = (v_sa_tokens + v_xa_tokens) * 0.5
        #! only self attention
        a_tokens = a_sa_tokens
        v_tokens = v_sa_tokens
        #! only cross attention
        # a_tokens = a_xa_tokens
        # v_tokens = v_xa_tokens


        #! learnable, turn to nan
        # alpha_a = torch.sigmoid(self.alpha_a)
        # a_tokens = alpha_a * a_sa_tokens + (1 - alpha_a) * a_xa_tokens
        # alpha_v = torch.sigmoid(self.alpha_v)
        # v_tokens = alpha_v * v_sa_tokens + (1 - alpha_v) * v_xa_tokens
        # # pdb.set_trace()

        if  torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens


# ============================================================================
# Complex 1x1 Convolution for Q, K, V Projection
# ============================================================================

class Complex1x1Conv(nn.Module):
    """
    1x1 Complex Convolution layer.

    For complex input z = x + iy and complex weight W = A + iB:
    W * z = (Ax - By) + i(Ay + Bx)

    Uses Conv1d with kernel_size=1 for the projection.
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Complex1x1Conv, self).__init__()
        # Real and imaginary parts of the complex convolution weights
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, real, imag):
        """
        Args:
            real: [bs, seq_len, channels] - real part (from visual)
            imag: [bs, seq_len, channels] - imaginary part (from audio)
        Returns:
            out_real, out_imag: [bs, seq_len, out_channels]
        """
        # Conv1d expects [bs, channels, seq_len], so transpose
        real = real.transpose(1, 2)  # [bs, channels, seq_len]
        imag = imag.transpose(1, 2)

        # Complex convolution: (A + iB) * (x + iy) = (Ax - By) + i(Ay + Bx)
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)

        # Transpose back to [bs, seq_len, channels]
        out_real = out_real.transpose(1, 2)
        out_imag = out_imag.transpose(1, 2)

        return out_real, out_imag


class ComplexTemporalConv(nn.Module):
    """
    Temporal Complex Convolution layer with configurable kernel size.

    For complex input z = x + iy and complex weight W = A + iB:
    W * z = (Ax - By) + i(Ay + Bx)

    Uses Conv1d with kernel_size=5 (default) for temporal modeling.
    Padding maintains the same sequence length.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, bias=True):
        super(ComplexTemporalConv, self).__init__()
        self.kernel_size = kernel_size
        # Padding to maintain sequence length: (kernel_size - 1) // 2
        self.padding = (kernel_size - 1) // 2

        # Real and imaginary parts of the complex convolution weights
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=self.padding, bias=bias)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=self.padding, bias=bias)

    def forward(self, real, imag):
        """
        Args:
            real: [bs, seq_len, channels] - real part (from visual)
            imag: [bs, seq_len, channels] - imaginary part (from audio)
        Returns:
            out_real, out_imag: [bs, seq_len, out_channels]
        """
        # Conv1d expects [bs, channels, seq_len], so transpose
        real = real.transpose(1, 2)  # [bs, channels, seq_len]
        imag = imag.transpose(1, 2)

        # Complex convolution: (A + iB) * (x + iy) = (Ax - By) + i(Ay + Bx)
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)

        # Transpose back to [bs, seq_len, channels]
        out_real = out_real.transpose(1, 2)
        out_imag = out_imag.transpose(1, 2)

        return out_real, out_imag


class ComplexProjectionSelfAttentionModule(nn.Module):
    """
    Complex Projection Self-Attention Module.

    Architecture:
        1. Visual → nn.Linear → Q_real, K_real, V_real
        2. Audio  → nn.Linear → Q_imag, K_imag, V_imag
        3. Form complex Q, K, V: Q = Q_real + i*Q_imag (visual=real, audio=imag)
        4. 1x1 Complex Conv projection on Q, K, V → Q', K', V'
        5. Split into real and imaginary parts
        6. Real parts (Q'_r, K'_r, V'_r) → Real Self-Attention → Visual output
        7. Imag parts (Q'_i, K'_i, V'_i) → Real Self-Attention → Audio output

    The cross-modal fusion happens through the complex convolution projection.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None
    ):
        super(ComplexProjectionSelfAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.n_heads = head_num
        self.win_size = win_size
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn

        # Step 1: Linear projections for Visual → Q_real, K_real, V_real
        self.visual_q_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_k_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_v_proj = nn.Linear(feat_dim, hid_dim)

        # Step 2: Linear projections for Audio → Q_imag, K_imag, V_imag
        self.audio_q_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_k_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_v_proj = nn.Linear(feat_dim, hid_dim)

        # Step 4: 1x1 Complex Conv projections for Q, K, V
        self.complex_q_conv = Complex1x1Conv(hid_dim, hid_dim)
        self.complex_k_conv = Complex1x1Conv(hid_dim, hid_dim)
        self.complex_v_conv = Complex1x1Conv(hid_dim, hid_dim)

        # Step 6: Real Self-Attention for visual (real parts)
        self.visual_self_attn = MultiHeadAttention(hid_dim, hid_dim, hid_dim, head_num, norm_flag)
        self.visual_ffn = PositionwiseFeedForward(hid_dim, d_ff, dropout)
        self.visual_norm1 = nn.LayerNorm(hid_dim)
        self.visual_norm2 = nn.LayerNorm(hid_dim)

        # Step 7: Real Self-Attention for audio (imaginary parts)
        self.audio_self_attn = MultiHeadAttention(hid_dim, hid_dim, hid_dim, head_num, norm_flag)
        self.audio_ffn = PositionwiseFeedForward(hid_dim, d_ff, dropout)
        self.audio_norm1 = nn.LayerNorm(hid_dim)
        self.audio_norm2 = nn.LayerNorm(hid_dim)

        # Output projections to restore original dimension
        self.visual_out_proj = nn.Linear(hid_dim, feat_dim)
        self.audio_out_proj = nn.Linear(hid_dim, feat_dim)

        # Distance adjacency for temporal modeling
        self.loc_adj = DistanceAdj(gamma, bias)

    def forward(self, a_tokens, v_tokens):
        """
        Args:
            a_tokens: [bs, 10, 1024] - audio features
            v_tokens: [bs, 10, 1024] - visual features
        Returns:
            a_tokens: [bs, 10, 1024] - processed audio features
            v_tokens: [bs, 10, 1024] - processed visual features
        """
        bs = a_tokens.shape[0]
        seq_len = a_tokens.shape[1]

        # ============================================
        # Step 1: Visual → Q_real, K_real, V_real
        # ============================================
        Q_real = self.visual_q_proj(v_tokens)  # [bs, 10, hid_dim]
        K_real = self.visual_k_proj(v_tokens)  # [bs, 10, hid_dim]
        V_real = self.visual_v_proj(v_tokens)  # [bs, 10, hid_dim]

        # ============================================
        # Step 2: Audio → Q_imag, K_imag, V_imag
        # ============================================
        Q_imag = self.audio_q_proj(a_tokens)  # [bs, 10, hid_dim]
        K_imag = self.audio_k_proj(a_tokens)  # [bs, 10, hid_dim]
        V_imag = self.audio_v_proj(a_tokens)  # [bs, 10, hid_dim]

        # ============================================
        # Step 3: Form Complex Q, K, V (implicit)
        # Q = Q_real + i*Q_imag (visual=real, audio=imag)
        # ============================================

        # ============================================
        # Step 4: 1x1 Complex Conv projection
        # ============================================
        Q_real_prime, Q_imag_prime = self.complex_q_conv(Q_real, Q_imag)
        K_real_prime, K_imag_prime = self.complex_k_conv(K_real, K_imag)
        V_real_prime, V_imag_prime = self.complex_v_conv(V_real, V_imag)

        # ============================================
        # Step 5: Get adjacency matrix if needed
        # ============================================
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(bs, seq_len, self.n_heads)

        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, seq_len, bs)

        # ============================================
        # Step 6: Real Self-Attention on real parts → Visual output
        # ============================================
        v_attn_out = self.visual_self_attn(Q_real_prime, K_real_prime, V_real_prime, mask, adj)
        v_out = Q_real_prime + v_attn_out  # Residual connection
        v_out = self.visual_norm1(v_out)

        v_ffn_out = self.visual_ffn(v_out)
        v_out = v_out + v_ffn_out  # Residual connection
        v_out = self.visual_norm2(v_out)

        # ============================================
        # Step 7: Real Self-Attention on imaginary parts → Audio output
        # ============================================
        a_attn_out = self.audio_self_attn(Q_imag_prime, K_imag_prime, V_imag_prime, mask, adj)
        a_out = Q_imag_prime + a_attn_out  # Residual connection
        a_out = self.audio_norm1(a_out)

        a_ffn_out = self.audio_ffn(a_out)
        a_out = a_out + a_ffn_out  # Residual connection
        a_out = self.audio_norm2(a_out)

        # ============================================
        # Step 8: Output projection to restore dimension
        # ============================================
        v_tokens = self.visual_out_proj(v_out)  # [bs, 10, feat_dim]
        a_tokens = self.audio_out_proj(a_out)   # [bs, 10, feat_dim]

        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens

    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
        m = m.repeat(bs, self.n_heads, 1, 1).cuda()
        return m


class ComplexTemporalProjectionSelfAttentionModule(nn.Module):
    """
    Complex Temporal Projection Self-Attention Module with 5x1 kernel.

    Architecture:
        1. Visual → nn.Linear → Q_real, K_real, V_real
        2. Audio  → nn.Linear → Q_imag, K_imag, V_imag
        3. Form complex Q, K, V: Q = Q_real + i*Q_imag (visual=real, audio=imag)
        4. 5x1 Complex Conv projection on Q, K, V → Q', K', V' (temporal modeling)
        5. Split into real and imaginary parts
        6. Real parts (Q'_r, K'_r, V'_r) → Real Self-Attention → Visual output
        7. Imag parts (Q'_i, K'_i, V'_i) → Real Self-Attention → Audio output

    The cross-modal fusion happens through the complex temporal convolution projection.
    The 5x1 kernel captures temporal dependencies across 5 timesteps.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        temporal_kernel=5  # Configurable temporal kernel size
    ):
        super(ComplexTemporalProjectionSelfAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.n_heads = head_num
        self.win_size = win_size
        self.use_adj_in_attn = use_adj_in_attn
        self.use_mask_in_attn = use_mask_in_attn
        self.temporal_kernel = temporal_kernel

        # Step 1: Linear projections for Visual → Q_real, K_real, V_real
        self.visual_q_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_k_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_v_proj = nn.Linear(feat_dim, hid_dim)

        # Step 2: Linear projections for Audio → Q_imag, K_imag, V_imag
        self.audio_q_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_k_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_v_proj = nn.Linear(feat_dim, hid_dim)

        # Step 4: 5x1 Complex Temporal Conv projections for Q, K, V
        self.complex_q_conv = ComplexTemporalConv(hid_dim, hid_dim, kernel_size=temporal_kernel)
        self.complex_k_conv = ComplexTemporalConv(hid_dim, hid_dim, kernel_size=temporal_kernel)
        self.complex_v_conv = ComplexTemporalConv(hid_dim, hid_dim, kernel_size=temporal_kernel)

        # Step 6: Real Self-Attention for visual (real parts)
        self.visual_self_attn = MultiHeadAttention(hid_dim, hid_dim, hid_dim, head_num, norm_flag)
        self.visual_ffn = PositionwiseFeedForward(hid_dim, d_ff, dropout)
        self.visual_norm1 = nn.LayerNorm(hid_dim)
        self.visual_norm2 = nn.LayerNorm(hid_dim)

        # Step 7: Real Self-Attention for audio (imaginary parts)
        self.audio_self_attn = MultiHeadAttention(hid_dim, hid_dim, hid_dim, head_num, norm_flag)
        self.audio_ffn = PositionwiseFeedForward(hid_dim, d_ff, dropout)
        self.audio_norm1 = nn.LayerNorm(hid_dim)
        self.audio_norm2 = nn.LayerNorm(hid_dim)

        # Output projections to restore original dimension
        self.visual_out_proj = nn.Linear(hid_dim, feat_dim)
        self.audio_out_proj = nn.Linear(hid_dim, feat_dim)

        # Distance adjacency for temporal modeling
        self.loc_adj = DistanceAdj(gamma, bias)

    def forward(self, a_tokens, v_tokens):
        """
        Args:
            a_tokens: [bs, 10, 1024] - audio features
            v_tokens: [bs, 10, 1024] - visual features
        Returns:
            a_tokens: [bs, 10, 1024] - processed audio features
            v_tokens: [bs, 10, 1024] - processed visual features
        """
        bs = a_tokens.shape[0]
        seq_len = a_tokens.shape[1]

        # ============================================
        # Step 1: Visual → Q_real, K_real, V_real
        # ============================================
        Q_real = self.visual_q_proj(v_tokens)  # [bs, 10, hid_dim]
        K_real = self.visual_k_proj(v_tokens)  # [bs, 10, hid_dim]
        V_real = self.visual_v_proj(v_tokens)  # [bs, 10, hid_dim]

        # ============================================
        # Step 2: Audio → Q_imag, K_imag, V_imag
        # ============================================
        Q_imag = self.audio_q_proj(a_tokens)  # [bs, 10, hid_dim]
        K_imag = self.audio_k_proj(a_tokens)  # [bs, 10, hid_dim]
        V_imag = self.audio_v_proj(a_tokens)  # [bs, 10, hid_dim]

        # ============================================
        # Step 3: Form Complex Q, K, V (implicit)
        # Q = Q_real + i*Q_imag (visual=real, audio=imag)
        # ============================================

        # ============================================
        # Step 4: 5x1 Complex Temporal Conv projection
        # This captures temporal dependencies across 5 timesteps
        # ============================================
        Q_real_prime, Q_imag_prime = self.complex_q_conv(Q_real, Q_imag)
        K_real_prime, K_imag_prime = self.complex_k_conv(K_real, K_imag)
        V_real_prime, V_imag_prime = self.complex_v_conv(V_real, V_imag)

        # ============================================
        # Step 5: Get adjacency matrix if needed
        # ============================================
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(bs, seq_len, self.n_heads)

        mask = None
        if self.use_mask_in_attn:
            mask = self.get_mask(self.win_size, seq_len, bs)

        # ============================================
        # Step 6: Real Self-Attention on real parts → Visual output
        # ============================================
        v_attn_out = self.visual_self_attn(Q_real_prime, K_real_prime, V_real_prime, mask, adj)
        v_out = Q_real_prime + v_attn_out  # Residual connection
        v_out = self.visual_norm1(v_out)

        v_ffn_out = self.visual_ffn(v_out)
        v_out = v_out + v_ffn_out  # Residual connection
        v_out = self.visual_norm2(v_out)

        # ============================================
        # Step 7: Real Self-Attention on imaginary parts → Audio output
        # ============================================
        a_attn_out = self.audio_self_attn(Q_imag_prime, K_imag_prime, V_imag_prime, mask, adj)
        a_out = Q_imag_prime + a_attn_out  # Residual connection
        a_out = self.audio_norm1(a_out)

        a_ffn_out = self.audio_ffn(a_out)
        a_out = a_out + a_ffn_out  # Residual connection
        a_out = self.audio_norm2(a_out)

        # ============================================
        # Step 8: Output projection to restore dimension
        # ============================================
        v_tokens = self.visual_out_proj(v_out)  # [bs, 10, feat_dim]
        a_tokens = self.audio_out_proj(a_out)   # [bs, 10, feat_dim]

        if torch.sum(a_tokens.isnan()) > 0 or torch.sum(v_tokens.isnan()) > 0:
            pdb.set_trace()

        return a_tokens, v_tokens

    def get_mask(self, window_size, temporal_scale, bs):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
        m = m.repeat(bs, self.n_heads, 1, 1).cuda()
        return m


# ============================================================================
# Text-Headed Temporal Attention Module
# ============================================================================

class TextHeadedTemporalAttention(nn.Module):
    """
    Multi-head attention where each head corresponds to a text category.

    - Text embeddings become the Queries (one per category/head)
    - Audio/Visual features are Keys and Values
    - Each head learns to detect temporal patterns for its category

    This allows the model to learn category-specific temporal patterns,
    where each attention head specializes in finding relevant A/V frames
    for a particular event category.

    For open-vocabulary: Novel category text embeddings from ImageBind
    can still produce meaningful attention patterns since they share
    the same semantic embedding space.
    """
    def __init__(
        self,
        feat_dim=1024,
        text_dim=1024,
        hid_dim=256,
        dropout=0.1,
        use_complex_fusion=True,
        num_refinement_layers=1
    ):
        super(TextHeadedTemporalAttention, self).__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.hid_dim = hid_dim
        self.use_complex_fusion = use_complex_fusion
        self.scale = hid_dim ** -0.5

        # Project text embeddings to queries
        self.text_q_proj = nn.Linear(text_dim, hid_dim)

        # Project Audio to keys and values
        self.audio_k_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_v_proj = nn.Linear(feat_dim, hid_dim)

        # Project Visual to keys and values
        self.visual_k_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_v_proj = nn.Linear(feat_dim, hid_dim)

        # Learnable gating for A/V combination
        self.av_gate = nn.Parameter(torch.ones(2) * 0.5)

        # Output projections
        self.out_proj_audio = nn.Linear(hid_dim, feat_dim)
        self.out_proj_visual = nn.Linear(hid_dim, feat_dim)

        # Layer norms
        self.norm_audio = nn.LayerNorm(feat_dim)
        self.norm_visual = nn.LayerNorm(feat_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Optional: Additional refinement with standard temporal self-attention
        self.num_refinement_layers = num_refinement_layers
        if num_refinement_layers > 0:
            self.audio_refinement = nn.ModuleList([
                TransformerEncoder(
                    d_model=feat_dim,
                    hid_dim=hid_dim,
                    d_ff=hid_dim * 2,
                    n_heads=4,
                    dropout=dropout
                ) for _ in range(num_refinement_layers)
            ])
            self.visual_refinement = nn.ModuleList([
                TransformerEncoder(
                    d_model=feat_dim,
                    hid_dim=hid_dim,
                    d_ff=hid_dim * 2,
                    n_heads=4,
                    dropout=dropout
                ) for _ in range(num_refinement_layers)
            ])

    def forward(self, audio, visual, text_embeddings):
        """
        Args:
            audio: [bs, T, D] - audio features per frame (T=10, D=1024)
            visual: [bs, T, D] - visual features per frame
            text_embeddings: [C, D_text] - category text embeddings (C=num_categories)

        Returns:
            audio_out: [bs, T, D] - refined audio features
            visual_out: [bs, T, D] - refined visual features
            attn_weights: [bs, C, T] - per-category temporal attention weights
        """
        bs, T, D = audio.shape
        C = text_embeddings.shape[0]  # num categories

        # ============================================
        # Step 1: Project text to queries [C, hid_dim]
        # ============================================
        Q = self.text_q_proj(text_embeddings)  # [C, hid_dim]

        # ============================================
        # Step 2: Project A/V to keys and values
        # ============================================
        K_a = self.audio_k_proj(audio)    # [bs, T, hid_dim]
        V_a = self.audio_v_proj(audio)    # [bs, T, hid_dim]
        K_v = self.visual_k_proj(visual)  # [bs, T, hid_dim]
        V_v = self.visual_v_proj(visual)  # [bs, T, hid_dim]

        # ============================================
        # Step 3: Combine A/V keys and values
        # ============================================
        gate = F.softmax(self.av_gate, dim=0)

        if self.use_complex_fusion:
            # Complex-style: treat A as real, V as imaginary
            # Attention on magnitude-like combination
            K = torch.sqrt(K_a ** 2 + K_v ** 2 + 1e-8)  # [bs, T, hid_dim]
            V_combined = gate[0] * V_a + gate[1] * V_v
        else:
            # Simple gated combination
            K = gate[0] * K_a + gate[1] * K_v  # [bs, T, hid_dim]
            V_combined = gate[0] * V_a + gate[1] * V_v

        # ============================================
        # Step 4: Compute attention per category head
        # ============================================
        # Q: [C, hid_dim], K: [bs, T, hid_dim]
        # attn_scores: [bs, C, T]
        attn_scores = torch.einsum('cd,btd->bct', Q, K) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, C, T]
        attn_weights_dropped = self.dropout(attn_weights)

        # ============================================
        # Step 5: Compute category-aware outputs
        # ============================================
        # attn_weights: [bs, C, T], V_combined: [bs, T, hid_dim]
        # category_out: [bs, C, hid_dim]
        category_out = torch.einsum('bct,btd->bcd', attn_weights_dropped, V_combined)

        # ============================================
        # Step 6: Broadcast back to temporal dimension
        # ============================================
        # Use transposed attention to get frame-level refinements
        # Each frame gets weighted contributions from all category heads

        # frame_category_weights: [bs, T, C] - how much each frame relates to each category
        frame_category_weights = attn_weights_dropped.transpose(1, 2)  # [bs, T, C]
        frame_category_weights = F.softmax(frame_category_weights, dim=-1)

        # Weighted combination of category outputs per frame
        # [bs, T, C] @ [bs, C, hid_dim] -> [bs, T, hid_dim]
        temporal_refinement = torch.einsum('btc,bcd->btd', frame_category_weights, category_out)

        # ============================================
        # Step 7: Project and apply residual
        # ============================================
        audio_refinement = self.out_proj_audio(temporal_refinement)
        visual_refinement = self.out_proj_visual(temporal_refinement)

        audio_out = self.norm_audio(audio + audio_refinement)
        visual_out = self.norm_visual(visual + visual_refinement)

        # ============================================
        # Step 8: Optional temporal self-attention refinement
        # ============================================
        if self.num_refinement_layers > 0:
            for audio_layer, visual_layer in zip(self.audio_refinement, self.visual_refinement):
                audio_out = audio_layer(audio_out, audio_out, audio_out)
                visual_out = visual_layer(visual_out, visual_out, visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out, attn_weights


class TextHeadedComplexTemporalAttention(nn.Module):
    """
    Complex-valued version of Text-Headed Temporal Attention.

    - Text embeddings become Queries
    - Audio features are the REAL component of Keys/Values
    - Visual features are the IMAGINARY component of Keys/Values
    - Complex attention naturally fuses A/V through complex arithmetic

    Complex attention score: Re(Q @ K^H) where K = K_audio + i*K_visual
    This creates cross-modal interaction through the complex conjugate transpose.
    """
    def __init__(
        self,
        feat_dim=1024,
        text_dim=1024,
        hid_dim=256,
        dropout=0.1,
        num_refinement_layers=1
    ):
        super(TextHeadedComplexTemporalAttention, self).__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.hid_dim = hid_dim
        self.scale = hid_dim ** -0.5

        # Project text embeddings to complex queries (real and imag parts)
        self.text_q_real_proj = nn.Linear(text_dim, hid_dim)
        self.text_q_imag_proj = nn.Linear(text_dim, hid_dim)

        # Project Audio to keys and values (REAL component)
        self.audio_k_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_v_proj = nn.Linear(feat_dim, hid_dim)

        # Project Visual to keys and values (IMAGINARY component)
        self.visual_k_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_v_proj = nn.Linear(feat_dim, hid_dim)

        # Output projections
        self.out_proj_audio = nn.Linear(hid_dim, feat_dim)
        self.out_proj_visual = nn.Linear(hid_dim, feat_dim)

        # Layer norms
        self.norm_audio = nn.LayerNorm(feat_dim)
        self.norm_visual = nn.LayerNorm(feat_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Optional refinement layers
        self.num_refinement_layers = num_refinement_layers
        if num_refinement_layers > 0:
            self.refinement = nn.ModuleList([
                ComplexTransformerEncoder(
                    d_model=feat_dim,
                    hid_dim=hid_dim,
                    d_ff=hid_dim * 2,
                    n_heads=4,
                    dropout=dropout
                ) for _ in range(num_refinement_layers)
            ])

    def forward(self, audio, visual, text_embeddings):
        """
        Args:
            audio: [bs, T, D] - audio features (REAL component)
            visual: [bs, T, D] - visual features (IMAGINARY component)
            text_embeddings: [C, D_text] - category text embeddings

        Returns:
            audio_out: [bs, T, D] - refined audio features
            visual_out: [bs, T, D] - refined visual features
            attn_weights: [bs, C, T] - per-category temporal attention weights
        """
        bs, T, D = audio.shape
        C = text_embeddings.shape[0]

        # ============================================
        # Step 1: Project text to complex queries
        # ============================================
        Q_real = self.text_q_real_proj(text_embeddings)  # [C, hid_dim]
        Q_imag = self.text_q_imag_proj(text_embeddings)  # [C, hid_dim]

        # ============================================
        # Step 2: Project A/V to complex keys and values
        # ============================================
        # Audio = Real part, Visual = Imaginary part
        K_real = self.audio_k_proj(audio)    # [bs, T, hid_dim]
        K_imag = self.visual_k_proj(visual)  # [bs, T, hid_dim]
        V_real = self.audio_v_proj(audio)    # [bs, T, hid_dim]
        V_imag = self.visual_v_proj(visual)  # [bs, T, hid_dim]

        # ============================================
        # Step 3: Complex attention scores
        # ============================================
        # Q @ K^H where K^H is conjugate transpose
        # (Q_r + i*Q_i) @ (K_r - i*K_i)^T = (Q_r @ K_r^T + Q_i @ K_i^T) + i*(Q_i @ K_r^T - Q_r @ K_i^T)
        # We use the REAL part for attention weights (common practice)

        # Q: [C, hid_dim], K: [bs, T, hid_dim]
        # attn_real: [bs, C, T]
        attn_real = (torch.einsum('cd,btd->bct', Q_real, K_real) +
                     torch.einsum('cd,btd->bct', Q_imag, K_imag))
        attn_real = attn_real * self.scale

        attn_weights = F.softmax(attn_real, dim=-1)  # [bs, C, T]
        attn_weights_dropped = self.dropout(attn_weights)

        # ============================================
        # Step 4: Apply attention to complex values
        # ============================================
        # category_out_real: [bs, C, hid_dim]
        category_out_real = torch.einsum('bct,btd->bcd', attn_weights_dropped, V_real)
        category_out_imag = torch.einsum('bct,btd->bcd', attn_weights_dropped, V_imag)

        # ============================================
        # Step 5: Broadcast back to temporal dimension
        # ============================================
        frame_category_weights = attn_weights_dropped.transpose(1, 2)  # [bs, T, C]
        frame_category_weights = F.softmax(frame_category_weights, dim=-1)

        # temporal_refinement_real/imag: [bs, T, hid_dim]
        temporal_refinement_real = torch.einsum('btc,bcd->btd', frame_category_weights, category_out_real)
        temporal_refinement_imag = torch.einsum('btc,bcd->btd', frame_category_weights, category_out_imag)

        # ============================================
        # Step 6: Project and apply residual
        # ============================================
        # Real -> Audio, Imag -> Visual
        audio_refinement = self.out_proj_audio(temporal_refinement_real)
        visual_refinement = self.out_proj_visual(temporal_refinement_imag)

        audio_out = self.norm_audio(audio + audio_refinement)
        visual_out = self.norm_visual(visual + visual_refinement)

        # ============================================
        # Step 7: Optional complex self-attention refinement
        # ============================================
        if self.num_refinement_layers > 0:
            for layer in self.refinement:
                audio_out, visual_out = layer(audio_out, visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out, attn_weights


class TextHeadedCrossAttention(nn.Module):
    """
    True cross-attention where text queries attend to audio and visual SEPARATELY.

    - Text embeddings → Queries (one per category)
    - Audio features → K_a, V_a (separate)
    - Visual features → K_v, V_v (separate)

    Each category attends to audio and visual independently.
    Output: refined audio and visual features.

    Then compute:
        sim_a = refined_audio @ text.T  → real part
        sim_v = refined_visual @ text.T → imaginary part
    And pass to ComplexCNN for fusion.
    """
    def __init__(
        self,
        feat_dim=1024,
        text_dim=1024,
        hid_dim=256,
        dropout=0.1,
        num_refinement_layers=1
    ):
        super(TextHeadedCrossAttention, self).__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.hid_dim = hid_dim
        self.scale = hid_dim ** -0.5

        # Project text embeddings to queries (shared for both modalities)
        self.text_q_proj = nn.Linear(text_dim, hid_dim)

        # Separate K, V projections for Audio
        self.audio_k_proj = nn.Linear(feat_dim, hid_dim)
        self.audio_v_proj = nn.Linear(feat_dim, hid_dim)

        # Separate K, V projections for Visual
        self.visual_k_proj = nn.Linear(feat_dim, hid_dim)
        self.visual_v_proj = nn.Linear(feat_dim, hid_dim)

        # Output projections back to feat_dim
        self.out_proj_audio = nn.Linear(hid_dim, feat_dim)
        self.out_proj_visual = nn.Linear(hid_dim, feat_dim)

        # Layer norms
        self.norm_audio = nn.LayerNorm(feat_dim)
        self.norm_visual = nn.LayerNorm(feat_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Optional: Refinement with temporal self-attention
        self.num_refinement_layers = num_refinement_layers
        if num_refinement_layers > 0:
            self.audio_refinement = nn.ModuleList([
                TransformerEncoder(
                    d_model=feat_dim,
                    hid_dim=hid_dim,
                    d_ff=hid_dim * 2,
                    n_heads=4,
                    dropout=dropout
                ) for _ in range(num_refinement_layers)
            ])
            self.visual_refinement = nn.ModuleList([
                TransformerEncoder(
                    d_model=feat_dim,
                    hid_dim=hid_dim,
                    d_ff=hid_dim * 2,
                    n_heads=4,
                    dropout=dropout
                ) for _ in range(num_refinement_layers)
            ])

    def forward(self, audio, visual, text_embeddings):
        """
        Args:
            audio: [bs, T, D] - audio features per frame
            visual: [bs, T, D] - visual features per frame
            text_embeddings: [C, D_text] - category text embeddings

        Returns:
            audio_out: [bs, T, D] - refined audio features
            visual_out: [bs, T, D] - refined visual features

        After this, compute similarity and pass to ComplexCNN:
            sim_a = F.normalize(audio_out, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
            sim_v = F.normalize(visual_out, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
            fused = complex_cnn(sim_v, sim_a)  # real=visual, imag=audio (or vice versa)
        """
        bs, T, D = audio.shape
        C = text_embeddings.shape[0]  # num categories

        # ============================================
        # Step 1: Project text to queries [C, hid_dim]
        # ============================================
        Q = self.text_q_proj(text_embeddings)  # [C, hid_dim]

        # ============================================
        # Step 2: Project Audio to K_a, V_a
        # ============================================
        K_a = self.audio_k_proj(audio)    # [bs, T, hid_dim]
        V_a = self.audio_v_proj(audio)    # [bs, T, hid_dim]

        # ============================================
        # Step 3: Project Visual to K_v, V_v
        # ============================================
        K_v = self.visual_k_proj(visual)  # [bs, T, hid_dim]
        V_v = self.visual_v_proj(visual)  # [bs, T, hid_dim]

        # ============================================
        # Step 4: Cross-attention: Text → Audio
        # ============================================
        # Q: [C, hid_dim], K_a: [bs, T, hid_dim]
        audio_attn_scores = torch.einsum('cd,btd->bct', Q, K_a) * self.scale
        audio_attn_weights = F.softmax(audio_attn_scores, dim=-1)  # [bs, C, T]
        audio_attn_dropped = self.dropout(audio_attn_weights)

        # [bs, C, T] @ [bs, T, hid_dim] -> [bs, C, hid_dim]
        audio_category_out = torch.einsum('bct,btd->bcd', audio_attn_dropped, V_a)

        # ============================================
        # Step 5: Cross-attention: Text → Visual
        # ============================================
        visual_attn_scores = torch.einsum('cd,btd->bct', Q, K_v) * self.scale
        visual_attn_weights = F.softmax(visual_attn_scores, dim=-1)  # [bs, C, T]
        visual_attn_dropped = self.dropout(visual_attn_weights)

        # [bs, C, T] @ [bs, T, hid_dim] -> [bs, C, hid_dim]
        visual_category_out = torch.einsum('bct,btd->bcd', visual_attn_dropped, V_v)

        # ============================================
        # Step 6: Broadcast back to temporal dimension
        # ============================================
        # Each frame gets weighted contributions from category heads
        # frame_weights: [bs, T, C]
        audio_frame_weights = audio_attn_dropped.transpose(1, 2)  # [bs, T, C]
        audio_frame_weights = F.softmax(audio_frame_weights, dim=-1)

        visual_frame_weights = visual_attn_dropped.transpose(1, 2)  # [bs, T, C]
        visual_frame_weights = F.softmax(visual_frame_weights, dim=-1)

        # [bs, T, C] @ [bs, C, hid_dim] -> [bs, T, hid_dim]
        audio_temporal = torch.einsum('btc,bcd->btd', audio_frame_weights, audio_category_out)
        visual_temporal = torch.einsum('btc,bcd->btd', visual_frame_weights, visual_category_out)

        # ============================================
        # Step 7: Project and apply residual
        # ============================================
        audio_refinement = self.out_proj_audio(audio_temporal)
        visual_refinement = self.out_proj_visual(visual_temporal)

        audio_out = self.norm_audio(audio + audio_refinement)
        visual_out = self.norm_visual(visual + visual_refinement)

        # ============================================
        # Step 8: Optional temporal self-attention refinement
        # ============================================
        if self.num_refinement_layers > 0:
            for audio_layer, visual_layer in zip(self.audio_refinement, self.visual_refinement):
                audio_out = audio_layer(audio_out, audio_out, audio_out)
                visual_out = visual_layer(visual_out, visual_out, visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class DualComplexSelfAttentionWithTextImag(nn.Module):
    """
    Dual Complex Self-Attention with Text as Shared Imaginary Component.

    Text embeddings serve as a shared imaginary component paired with each modality
    as the real component. Two parallel complex-valued self-attention layers process
    (audio + i*text) and (visual + i*text) independently.

    Complex arithmetic allows text semantics to modulate temporal attention patterns
    through phase interactions. The real part of the output captures text-guided
    refinement of each modality.

    This design treats text as a semantic "prior" that influences how audio and visual
    features attend to themselves temporally, while keeping modality-specific processing
    separate.

    Flow:
        1. Broadcast text to temporal dimension: [C, D] -> [bs, T, D]
        2. Complex self-attention on (audio + i*text) -> refined_audio
        3. Complex self-attention on (visual + i*text) -> refined_visual
        4. Compute sim_a, sim_v and pass to ComplexCNN for fusion
    """
    def __init__(
        self,
        feat_dim=1024,
        text_dim=1024,
        hid_dim=256,
        d_ff=512,
        n_heads=4,
        num_layers=1,
        dropout=0.1,
        text_broadcast='attention',  # 'attention', 'mean', 'project'
        output_mode='magnitude'  # 'real', 'magnitude', 'concat', 'gate'
    ):
        super(DualComplexSelfAttentionWithTextImag, self).__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.hid_dim = hid_dim
        self.text_broadcast = text_broadcast
        self.output_mode = output_mode

        # Project text to feat_dim if different
        if text_dim != feat_dim:
            self.text_proj = nn.Linear(text_dim, feat_dim)
        else:
            self.text_proj = nn.Identity()

        # Text-to-temporal attention for broadcasting text to T frames
        if text_broadcast == 'attention':
            # Learn which text categories are relevant per frame
            self.text_temporal_attn = nn.Sequential(
                nn.Linear(feat_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, feat_dim)
            )
            # Query from modality, Key from text
            self.audio_text_query = nn.Linear(feat_dim, hid_dim)
            self.visual_text_query = nn.Linear(feat_dim, hid_dim)
            self.text_key = nn.Linear(feat_dim, hid_dim)
            self.scale = hid_dim ** -0.5

        # Complex self-attention layers for audio + i*text
        self.audio_complex_layers = nn.ModuleList([
            ComplexTransformerEncoder(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Complex self-attention layers for visual + i*text
        self.visual_complex_layers = nn.ModuleList([
            ComplexTransformerEncoder(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Output projections based on output_mode
        if output_mode == 'concat':
            # Concatenate real and imag, then project back to feat_dim
            self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
            self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)
        elif output_mode == 'gate':
            # Learnable gate between real and imag
            self.audio_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )
            self.visual_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )

        # Output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)

    def broadcast_text_to_temporal(self, modality_feat, text_embeddings):
        """
        Broadcast text embeddings [C, D] to temporal dimension [bs, T, D].

        Args:
            modality_feat: [bs, T, D] - audio or visual features
            text_embeddings: [C, D] - category text embeddings

        Returns:
            text_temporal: [bs, T, D] - text broadcasted to temporal dimension
        """
        bs, T, D = modality_feat.shape
        text_proj = self.text_proj(text_embeddings)  # [C, D]

        if self.text_broadcast == 'mean':
            # Simple mean over categories, expand to all frames
            text_mean = text_proj.mean(dim=0, keepdim=True)  # [1, D]
            text_temporal = text_mean.unsqueeze(0).expand(bs, T, -1)  # [bs, T, D]

        elif self.text_broadcast == 'attention':
            # Each frame attends to text categories to get frame-specific text
            if modality_feat is self._current_audio:
                Q = self.audio_text_query(modality_feat)  # [bs, T, hid_dim]
            else:
                Q = self.visual_text_query(modality_feat)  # [bs, T, hid_dim]

            K = self.text_key(text_proj)  # [C, hid_dim]

            # Attention: [bs, T, hid_dim] @ [hid_dim, C] -> [bs, T, C]
            attn_scores = torch.einsum('btd,cd->btc', Q, K) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, T, C]
            attn_weights = self.dropout(attn_weights)

            # Weighted sum of text: [bs, T, C] @ [C, D] -> [bs, T, D]
            text_temporal = torch.einsum('btc,cd->btd', attn_weights, text_proj)

        elif self.text_broadcast == 'project':
            # Learn a projection from C categories to T frames
            # Simple: just tile and let the model figure it out
            text_mean = text_proj.mean(dim=0, keepdim=True)  # [1, D]
            text_temporal = text_mean.unsqueeze(0).expand(bs, T, -1)  # [bs, T, D]
            text_temporal = self.text_temporal_attn(text_temporal)

        return text_temporal

    def forward(self, audio, visual, text_embeddings):
        """
        Args:
            audio: [bs, T, D] - audio features per frame
            visual: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - category text embeddings

        Returns:
            audio_out: [bs, T, D] - refined audio features
            visual_out: [bs, T, D] - refined visual features

        After this, compute similarity and pass to ComplexCNN:
            sim_a = F.normalize(audio_out) @ F.normalize(text_embeddings).T
            sim_v = F.normalize(visual_out) @ F.normalize(text_embeddings).T
            fused = complex_cnn(sim_v, sim_a)
        """
        # Store references for broadcast_text_to_temporal
        self._current_audio = audio

        # ============================================
        # Step 1: Broadcast text to temporal dimension
        # ============================================
        # Get text as imaginary component for each modality
        self._current_audio = audio
        text_for_audio = self.broadcast_text_to_temporal(audio, text_embeddings)  # [bs, T, D]

        self._current_audio = None  # Switch to visual
        text_for_visual = self.broadcast_text_to_temporal(visual, text_embeddings)  # [bs, T, D]

        # ============================================
        # Step 2: Complex self-attention for audio + i*text
        # ============================================
        # real = audio, imag = text
        audio_real = audio
        audio_imag = text_for_audio

        for layer in self.audio_complex_layers:
            audio_real, audio_imag = layer(audio_real, audio_imag)

        # ============================================
        # Step 3: Complex self-attention for visual + i*text
        # ============================================
        # real = visual, imag = text
        visual_real = visual
        visual_imag = text_for_visual

        for layer in self.visual_complex_layers:
            visual_real, visual_imag = layer(visual_real, visual_imag)

        # ============================================
        # Step 4: Extract refined features (complex → real)
        # ============================================
        if self.output_mode == 'real':
            # Use only real part (text-modulated modality features)
            audio_out = audio_real
            visual_out = visual_real

        elif self.output_mode == 'magnitude':
            # Magnitude: sqrt(real^2 + imag^2) - preserves energy from both
            audio_out = torch.sqrt(audio_real**2 + audio_imag**2 + 1e-8)
            visual_out = torch.sqrt(visual_real**2 + visual_imag**2 + 1e-8)

        elif self.output_mode == 'concat':
            # Concatenate real and imag, project back to feat_dim
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_out = self.audio_out_proj(audio_concat)
            visual_out = self.visual_out_proj(visual_concat)

        elif self.output_mode == 'gate':
            # Learnable gate between real and imag
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_g = self.audio_gate(audio_concat)
            visual_g = self.visual_gate(visual_concat)
            audio_out = audio_g * audio_real + (1 - audio_g) * audio_imag
            visual_out = visual_g * visual_real + (1 - visual_g) * visual_imag

        # Apply layer norm
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class IndependentComplexAttention(nn.Module):
    """
    Complex Multi-Head Attention with INDEPENDENT projections.

    Unlike ComplexLinear which couples weights via:
        out_real = W_r @ real - W_i @ imag  (coupled)
        out_imag = W_r @ imag + W_i @ real  (coupled)

    This uses independent projections:
        Q_real = W_q_real @ x  (independent)
        Q_imag = W_q_imag @ x  (independent)

    Then computes complex attention:
        attn = softmax(Re(Q @ K*) / sqrt(d))  where K* is conjugate
             = softmax((Q_r @ K_r.T + Q_i @ K_i.T) / sqrt(d))
        out_real = attn @ V_real
        out_imag = attn @ V_imag
    """
    def __init__(self, d_model, hid_dim, n_heads=1, dropout=0.1):
        super(IndependentComplexAttention, self).__init__()
        self.d_model = d_model
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # Independent Q, K, V projections for real part
        self.q_real = nn.Linear(d_model, hid_dim)
        self.k_real = nn.Linear(d_model, hid_dim)
        self.v_real = nn.Linear(d_model, hid_dim)

        # Independent Q, K, V projections for imaginary part
        self.q_imag = nn.Linear(d_model, hid_dim)
        self.k_imag = nn.Linear(d_model, hid_dim)
        self.v_imag = nn.Linear(d_model, hid_dim)

        # Output projections (also independent)
        self.o_real = nn.Linear(hid_dim, d_model)
        self.o_imag = nn.Linear(hid_dim, d_model)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [bs, seq_len, d_model] - input features (same for real and imag)
        Returns:
            out_real, out_imag: [bs, seq_len, d_model]
        """
        bs, seq_len, _ = x.shape

        # Independent projections from the SAME input
        Q_real = self.q_real(x)  # [bs, seq, hid_dim]
        K_real = self.k_real(x)
        V_real = self.v_real(x)

        Q_imag = self.q_imag(x)  # [bs, seq, hid_dim]
        K_imag = self.k_imag(x)
        V_imag = self.v_imag(x)

        # Reshape for multi-head: [bs, seq, hid] -> [bs, heads, seq, head_dim]
        Q_real = Q_real.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_real = K_real.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_real = V_real.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        Q_imag = Q_imag.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_imag = K_imag.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_imag = V_imag.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Complex attention score: Re(Q @ K*)
        # Q @ K* = (Q_r + i*Q_i)(K_r - i*K_i) = (Q_r*K_r + Q_i*K_i) + i*(Q_i*K_r - Q_r*K_i)
        # We use the real part: Q_r @ K_r.T + Q_i @ K_i.T
        attn_scores = (torch.matmul(Q_real, K_real.transpose(-2, -1)) +
                       torch.matmul(Q_imag, K_imag.transpose(-2, -1))) / self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out_real = torch.matmul(attn_weights, V_real)  # [bs, heads, seq, head_dim]
        out_imag = torch.matmul(attn_weights, V_imag)

        # Reshape back: [bs, heads, seq, head_dim] -> [bs, seq, hid_dim]
        out_real = out_real.transpose(1, 2).contiguous().view(bs, seq_len, self.hid_dim)
        out_imag = out_imag.transpose(1, 2).contiguous().view(bs, seq_len, self.hid_dim)

        # Output projection
        out_real = self.o_real(out_real)
        out_imag = self.o_imag(out_imag)

        return out_real, out_imag


class IndependentComplexTransformerEncoder(nn.Module):
    """
    Transformer Encoder using IndependentComplexAttention.

    Takes a single input x and processes it through complex attention
    with independent real/imag projections.
    """
    def __init__(self, d_model=1024, hid_dim=256, d_ff=512, n_heads=1, dropout=0.1):
        super(IndependentComplexTransformerEncoder, self).__init__()

        self.attn = IndependentComplexAttention(d_model, hid_dim, n_heads, dropout)

        # FFN for real and imaginary (independent)
        self.ffn_real = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_imag = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1_real = nn.LayerNorm(d_model)
        self.norm1_imag = nn.LayerNorm(d_model)
        self.norm2_real = nn.LayerNorm(d_model)
        self.norm2_imag = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [bs, seq_len, d_model] - input (used for both real and imag projections)
        Returns:
            out_real, out_imag: [bs, seq_len, d_model]
        """
        # Complex attention
        attn_real, attn_imag = self.attn(x)

        # Residual + norm (real part uses original x as residual)
        real = self.norm1_real(x + attn_real)
        imag = self.norm1_imag(attn_imag)  # No residual for imag (starts from 0)

        # FFN
        ffn_real = self.ffn_real(real)
        ffn_imag = self.ffn_imag(imag)

        # Residual + norm
        out_real = self.norm2_real(real + ffn_real)
        out_imag = self.norm2_imag(imag + ffn_imag)

        return out_real, out_imag


class IdentityComplexSelfAttention(nn.Module):
    """
    Identity Complex Self-Attention Ablation Module with INDEPENDENT projections.

    This tests whether complex attention structure helps when real and imaginary
    parts come from the SAME input, but use INDEPENDENT projection weights.

    Unlike the coupled ComplexLinear approach, this uses:
        Q_real = W_q_real @ x   (independent weight)
        Q_imag = W_q_imag @ x   (independent weight, different from W_q_real)

    This is a fair comparison to real-valued attention since:
    - Both have independent learnable projections
    - Complex version just uses complex attention formula
    """
    def __init__(
        self,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        n_heads=1,
        num_layers=1,
        dropout=0.1,
        output_mode='magnitude'  # 'real', 'magnitude', 'concat', 'gate'
    ):
        super(IdentityComplexSelfAttention, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.output_mode = output_mode
        self.num_layers = num_layers

        # Independent complex transformer layers for audio
        self.audio_layers = nn.ModuleList([
            IndependentComplexTransformerEncoder(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Independent complex transformer layers for visual
        self.visual_layers = nn.ModuleList([
            IndependentComplexTransformerEncoder(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Output projections based on output_mode
        if output_mode == 'concat':
            self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
            self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)
        elif output_mode == 'gate':
            self.audio_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )
            self.visual_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )

        # Output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def forward(self, audio, visual, text_embeddings=None):
        """
        Args:
            audio: [bs, T, D] - audio features per frame
            visual: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            audio_out: [bs, T, D] - refined audio features
            visual_out: [bs, T, D] - refined visual features
        """
        # Process audio through independent complex layers
        audio_real, audio_imag = audio, None
        for i, layer in enumerate(self.audio_layers):
            if i == 0:
                audio_real, audio_imag = layer(audio)
            else:
                # For subsequent layers, combine real+imag back and re-split
                # Or process real and imag through their respective paths
                combined = audio_real  # Use real as the main signal
                audio_real, audio_imag = layer(combined)

        # Process visual through independent complex layers
        visual_real, visual_imag = visual, None
        for i, layer in enumerate(self.visual_layers):
            if i == 0:
                visual_real, visual_imag = layer(visual)
            else:
                combined = visual_real
                visual_real, visual_imag = layer(combined)

        # Extract output based on mode
        if self.output_mode == 'real':
            audio_out = audio_real
            visual_out = visual_real

        elif self.output_mode == 'magnitude':
            audio_out = torch.sqrt(audio_real**2 + audio_imag**2 + 1e-8)
            visual_out = torch.sqrt(visual_real**2 + visual_imag**2 + 1e-8)

        elif self.output_mode == 'concat':
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_out = self.audio_out_proj(audio_concat)
            visual_out = self.visual_out_proj(visual_concat)

        elif self.output_mode == 'gate':
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_g = self.audio_gate(audio_concat)
            visual_g = self.visual_gate(visual_concat)
            audio_out = audio_g * audio_real + (1 - audio_g) * audio_imag
            visual_out = visual_g * visual_real + (1 - visual_g) * visual_imag

        # Apply layer norm
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class IdentityComplexTemporalAttentionModule(nn.Module):
    """
    Identity Complex Temporal Attention Module following the paper:
    "Building Blocks for a Complex-Valued Transformer Architecture" (ICASSP 2023)

    This module uses COUPLED complex projections (ComplexLinear) as per the paper,
    where W = W_r + i*W_i and the multiplication follows complex arithmetic:
        W @ z = (W_r @ z_r - W_i @ z_i) + i(W_r @ z_i + W_i @ z_r)

    For identity complex ablation: z = x + i*x (same input for real and imaginary)
    This tests whether the complex attention STRUCTURE helps even without
    cross-modal interaction between audio and visual.

    Structure matches TemporalAttentionModule:
        - Separate self-attention blocks for audio (a_sa_blocks)
        - Separate self-attention blocks for visual (v_sa_blocks)
        - Each block: ComplexMultiHeadAttention + ComplexFFN + ComplexLayerNorm

    Attention formula (Paper's Equation 8):
        attn_scores = softmax(Re(Q @ K*) / sqrt(d))
        where Re(Q @ K*) = Q_r @ K_r^T + Q_i @ K_i^T
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility with TemporalAttentionModule
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        output_mode='magnitude'  # 'real', 'magnitude', 'concat', 'gate'
    ):
        super(IdentityComplexTemporalAttentionModule, self).__init__()
        self.feat_dim = feat_dim
        self.output_mode = output_mode

        # Create ComplexTransformerEncoder block (uses ComplexLinear with coupled weights)
        block = ComplexTransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate self-attention blocks for audio and visual (matching TemporalAttentionModule)
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Output projections based on output_mode
        if output_mode == 'concat':
            self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
            self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)
        elif output_mode == 'gate':
            self.audio_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )
            self.visual_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid()
            )

        # Output layer norms (applied after output mode extraction)
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def forward(self, a_tokens, v_tokens, text_embeddings=None):
        """
        Args:
            a_tokens: [bs, T, D] - audio features per frame (T=10, D=1024)
            v_tokens: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            a_tokens: [bs, T, D] - refined audio features
            v_tokens: [bs, T, D] - refined visual features

        Process:
            1. For audio: create identity complex z = audio + i*audio
            2. Pass through audio self-attention blocks (ComplexTransformerEncoder)
            3. Extract output using output_mode (magnitude, real, concat, gate)
            4. Same process for visual with separate blocks
        """
        # =====================================================================
        # AUDIO STREAM: Identity complex z = audio + i*audio
        # =====================================================================
        # Initialize: real = audio, imag = audio (identity complex)
        audio_real, audio_imag = a_tokens, a_tokens

        # Pass through audio self-attention blocks
        for blk in self.a_sa_blocks:
            # ComplexTransformerEncoder expects (real, imag) and returns (real, imag)
            # It uses ComplexLinear which follows: W @ z = (W_r @ z_r - W_i @ z_i) + i(W_r @ z_i + W_i @ z_r)
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # =====================================================================
        # VISUAL STREAM: Identity complex z = visual + i*visual
        # =====================================================================
        # Initialize: real = visual, imag = visual (identity complex)
        visual_real, visual_imag = v_tokens, v_tokens

        # Pass through visual self-attention blocks
        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        # =====================================================================
        # OUTPUT MODE: Extract real-valued output from complex representation
        # =====================================================================
        if self.output_mode == 'real':
            # Just use the real part
            audio_out = audio_real
            visual_out = visual_real

        elif self.output_mode == 'magnitude':
            # Complex magnitude: |z| = sqrt(real^2 + imag^2)
            audio_out = torch.sqrt(audio_real**2 + audio_imag**2 + 1e-8)
            visual_out = torch.sqrt(visual_real**2 + visual_imag**2 + 1e-8)

        elif self.output_mode == 'concat':
            # Concatenate real and imag, then project back
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_out = self.audio_out_proj(audio_concat)
            visual_out = self.visual_out_proj(visual_concat)

        elif self.output_mode == 'gate':
            # Learnable gate to combine real and imag
            audio_concat = torch.cat([audio_real, audio_imag], dim=-1)
            visual_concat = torch.cat([visual_real, visual_imag], dim=-1)
            audio_g = self.audio_gate(audio_concat)
            visual_g = self.visual_gate(visual_concat)
            audio_out = audio_g * audio_real + (1 - audio_g) * audio_imag
            visual_out = visual_g * visual_real + (1 - visual_g) * visual_imag

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        # Apply final layer norm
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class SplitComplexTemporalAttentionModule(nn.Module):
    """
    Split Complex Temporal Attention Module.

    Instead of using identity complex (z = x + i*x), this module:
    1. SPLITS the 1024-dim feature into two 512-dim halves
    2. Uses first half as REAL, second half as IMAGINARY
    3. Processes through complex transformer (512-dim each)
    4. CONCATENATES real and imaginary outputs back to 1024-dim

    This approach:
    - Creates genuine heterogeneity between real and imaginary parts
    - Preserves ALL information (no discarding imaginary like 'real' mode)
    - Maintains same input/output dimensionality (1024 -> 1024)
    - Complex operations MIX the two halves through complex arithmetic:
        real_out = A*x_first - B*x_second
        imag_out = A*x_second + B*x_first

    Similar conceptually to how GLU/SwiGLU splits features for gating.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,   # Full feature dimension (will be split to feat_dim/2)
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
    ):
        super(SplitComplexTemporalAttentionModule, self).__init__()

        assert feat_dim % 2 == 0, f"feat_dim must be even for split, got {feat_dim}"
        self.feat_dim = feat_dim
        self.split_dim = feat_dim // 2  # 512 for 1024-dim input

        # Create ComplexTransformerEncoder with HALF the dimension
        # Since we split 1024 -> 512 + 512i, the complex operations work on 512-dim
        block = ComplexTransformerEncoder(
            d_model=self.split_dim,  # 512 instead of 1024
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate self-attention blocks for audio and visual
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Output layer norms (applied to the concatenated 1024-dim output)
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def forward(self, a_tokens, v_tokens, text_embeddings=None):
        """
        Args:
            a_tokens: [bs, T, D] - audio features per frame (T=10, D=1024)
            v_tokens: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            a_tokens: [bs, T, D] - refined audio features (1024-dim)
            v_tokens: [bs, T, D] - refined visual features (1024-dim)

        Process:
            1. Split audio 1024 -> 512 (real) + 512 (imag)
            2. Pass through audio self-attention blocks
            3. Concat real + imag -> 1024
            4. Same for visual with separate blocks
        """
        # =====================================================================
        # AUDIO STREAM: Split into real (first 512) and imag (second 512)
        # =====================================================================
        audio_real = a_tokens[:, :, :self.split_dim]   # [bs, T, 512]
        audio_imag = a_tokens[:, :, self.split_dim:]   # [bs, T, 512]

        # Pass through audio self-attention blocks
        for blk in self.a_sa_blocks:
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # Concatenate back to 1024-dim
        audio_out = torch.cat([audio_real, audio_imag], dim=-1)  # [bs, T, 1024]

        # =====================================================================
        # VISUAL STREAM: Split into real (first 512) and imag (second 512)
        # =====================================================================
        visual_real = v_tokens[:, :, :self.split_dim]  # [bs, T, 512]
        visual_imag = v_tokens[:, :, self.split_dim:]  # [bs, T, 512]

        # Pass through visual self-attention blocks
        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        # Concatenate back to 1024-dim
        visual_out = torch.cat([visual_real, visual_imag], dim=-1)  # [bs, T, 1024]

        # =====================================================================
        # Apply final layer norm + RESIDUAL CONNECTION
        # =====================================================================
        # Residual ensures original ImageBind features are preserved
        # Complex attention only adds refinement, never corrupts original
        audio_out = a_tokens + self.audio_norm(audio_out)
        visual_out = v_tokens + self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class ProjectedComplexTemporalAttentionModule(nn.Module):
    """
    Projected Complex Temporal Attention Module.

    Projects 1024-dim real features to complex representation using learned linear layers:
    1. Input: 1024-dim real features (audio or visual)
    2. Project: real_proj(x) -> real part (1024-dim)
                imag_proj(x) -> imaginary part (1024-dim)
    3. Process through ComplexTransformerEncoder (full 1024-dim)
    4. Output: Take real part (or magnitude) -> 1024-dim

    This approach:
    - Learns optimal real/imaginary decomposition from the data
    - Works on full 1024-dim (vs 512-dim in SplitComplex)
    - More expressive but more parameters than SplitComplex
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        output_mode='real',  # 'real', 'magnitude', or 'concat'
    ):
        super(ProjectedComplexTemporalAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.output_mode = output_mode

        # Projection layers: real features -> complex (real + imag)
        # Audio projections
        self.audio_real_proj = nn.Linear(feat_dim, feat_dim)
        self.audio_imag_proj = nn.Linear(feat_dim, feat_dim)

        # Visual projections
        self.visual_real_proj = nn.Linear(feat_dim, feat_dim)
        self.visual_imag_proj = nn.Linear(feat_dim, feat_dim)

        # Create ComplexTransformerEncoder with full dimension
        block = ComplexTransformerEncoder(
            d_model=feat_dim,  # Full 1024-dim
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate self-attention blocks for audio and visual
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Output projection if using concat mode (2048 -> 1024)
        if output_mode == 'concat':
            self.audio_out_proj = nn.Linear(2 * feat_dim, feat_dim)
            self.visual_out_proj = nn.Linear(2 * feat_dim, feat_dim)

        # Output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def forward(self, a_tokens, v_tokens, text_embeddings=None):
        """
        Args:
            a_tokens: [bs, T, D] - audio features per frame (T=10, D=1024)
            v_tokens: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            a_tokens: [bs, T, D] - refined audio features (1024-dim)
            v_tokens: [bs, T, D] - refined visual features (1024-dim)
        """
        # =====================================================================
        # AUDIO STREAM: Project to complex, process, extract output
        # =====================================================================
        audio_real = self.audio_real_proj(a_tokens)  # [bs, T, 1024]
        audio_imag = self.audio_imag_proj(a_tokens)  # [bs, T, 1024]

        # Pass through audio self-attention blocks
        for blk in self.a_sa_blocks:
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # Extract output based on mode
        if self.output_mode == 'real':
            audio_out = audio_real  # [bs, T, 1024]
        elif self.output_mode == 'magnitude':
            audio_out = torch.sqrt(audio_real ** 2 + audio_imag ** 2 + 1e-8)  # [bs, T, 1024]
        elif self.output_mode == 'concat':
            audio_out = self.audio_out_proj(torch.cat([audio_real, audio_imag], dim=-1))  # [bs, T, 1024]

        # =====================================================================
        # VISUAL STREAM: Project to complex, process, extract output
        # =====================================================================
        visual_real = self.visual_real_proj(v_tokens)  # [bs, T, 1024]
        visual_imag = self.visual_imag_proj(v_tokens)  # [bs, T, 1024]

        # Pass through visual self-attention blocks
        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        # Extract output based on mode
        if self.output_mode == 'real':
            visual_out = visual_real  # [bs, T, 1024]
        elif self.output_mode == 'magnitude':
            visual_out = torch.sqrt(visual_real ** 2 + visual_imag ** 2 + 1e-8)  # [bs, T, 1024]
        elif self.output_mode == 'concat':
            visual_out = self.visual_out_proj(torch.cat([visual_real, visual_imag], dim=-1))  # [bs, T, 1024]

        # =====================================================================
        # Apply final layer norm
        # =====================================================================
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class TemporalDiffComplexAttentionModule(nn.Module):
    """
    Temporal Difference Complex Attention Module.

    Uses temporal differences as the imaginary part of complex representation:
    - Real part: original frame features (static appearance/sound)
    - Imaginary part: difference from previous frame (motion/dynamics)

    For 10 frames [f0, f1, ..., f9]:
        real[t] = f[t]                    (what is in this frame)
        imag[t] = f[t] - f[t-1]           (what changed from previous frame)
        imag[0] = f[1] - f[0]             (first frame uses forward diff)

    For the last frame (t=9), padding options:
        'copy':     imag[9] = imag[8]     (copy previous difference)
        'zero':     imag[9] = 0           (no change information)
        'circular': imag[9] = f[9] - f[0] (wrap around)
        'reverse':  imag[9] = f[8] - f[9] (backward difference)

    After complex attention, concatenates real + imag back to original dimension.
    This preserves all information while allowing complex arithmetic to mix
    static features with temporal dynamics.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        padding_mode='copy',  # 'copy', 'zero', 'circular', 'reverse'
    ):
        super(TemporalDiffComplexAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.padding_mode = padding_mode

        # Create ComplexTransformerEncoder with FULL dimension
        # Since real=1024 and imag=1024 (temporal diff), complex ops work on 1024-dim
        block = ComplexTransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate self-attention blocks for audio and visual
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Output projection: concat(real, imag) -> feat_dim
        self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)

        # Output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def compute_temporal_diff(self, x):
        """
        Compute temporal differences for imaginary part.

        Args:
            x: [bs, T, D] - input features (T=10 frames)

        Returns:
            diff: [bs, T, D] - temporal differences
        """
        bs, T, D = x.shape

        # Compute consecutive differences: f[t] - f[t-1] for t=1..T-1
        diff = x[:, 1:, :] - x[:, :-1, :]  # [bs, T-1, D]

        # Handle first frame: use forward difference f[1] - f[0]
        first_diff = diff[:, 0:1, :]  # [bs, 1, D]

        # Handle last frame based on padding mode
        if self.padding_mode == 'copy':
            # Copy the (T-2)th difference (i.e., diff between frame T-2 and T-1)
            last_diff = diff[:, -1:, :]  # [bs, 1, D]
        elif self.padding_mode == 'zero':
            # No change information for last frame
            last_diff = torch.zeros(bs, 1, D, device=x.device, dtype=x.dtype)
        elif self.padding_mode == 'circular':
            # Wrap around: f[T-1] - f[0]
            last_diff = x[:, -1:, :] - x[:, 0:1, :]  # [bs, 1, D]
        elif self.padding_mode == 'reverse':
            # Backward difference: f[T-2] - f[T-1]
            last_diff = x[:, -2:-1, :] - x[:, -1:, :]  # [bs, 1, D]
        else:
            raise ValueError(f"Unknown padding_mode: {self.padding_mode}")

        # Construct full diff tensor: [first_diff, diff[1:], last_diff]
        # diff already has T-1 elements, we need T elements
        # Position 0: first_diff (forward diff for frame 0)
        # Position 1 to T-2: diff[0] to diff[T-3] (backward diff for frames 1 to T-2)
        # Position T-1: last_diff (handled by padding mode)

        # Actually, let's reconsider:
        # For frame t, we want imag[t] = f[t] - f[t-1] (what changed TO get to this frame)
        # For frame 0, there's no t-1, so we use forward diff: f[1] - f[0]
        # For frame T-1, the diff is f[T-1] - f[T-2] which is diff[-1]
        # So we just need to prepend first_diff to handle frame 0

        # diff = [f1-f0, f2-f1, ..., f9-f8] has 9 elements
        # We want: [f1-f0, f1-f0, f2-f1, ..., f9-f8]
        #          frame0  frame1  frame2      frame9
        # So frame 0 uses forward diff, frames 1-9 use backward diff

        full_diff = torch.cat([first_diff, diff], dim=1)  # [bs, T, D]

        return full_diff

    def forward(self, a_tokens, v_tokens, text_embeddings=None):
        """
        Args:
            a_tokens: [bs, T, D] - audio features per frame (T=10, D=1024)
            v_tokens: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            a_tokens: [bs, T, D] - refined audio features
            v_tokens: [bs, T, D] - refined visual features

        Process:
            1. Compute temporal differences for imaginary part
            2. real = original features, imag = temporal diff
            3. Pass through complex self-attention blocks
            4. Concat real + imag, project back to D
        """
        # =====================================================================
        # AUDIO STREAM
        # =====================================================================
        audio_real = a_tokens  # [bs, T, 1024] - static features
        audio_imag = self.compute_temporal_diff(a_tokens)  # [bs, T, 1024] - temporal dynamics

        # Pass through audio self-attention blocks
        for blk in self.a_sa_blocks:
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # Concatenate and project back to feat_dim
        audio_concat = torch.cat([audio_real, audio_imag], dim=-1)  # [bs, T, 2048]
        audio_out = self.audio_out_proj(audio_concat)  # [bs, T, 1024]

        # =====================================================================
        # VISUAL STREAM
        # =====================================================================
        visual_real = v_tokens  # [bs, T, 1024] - static features
        visual_imag = self.compute_temporal_diff(v_tokens)  # [bs, T, 1024] - temporal dynamics

        # Pass through visual self-attention blocks
        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        # Concatenate and project back to feat_dim
        visual_concat = torch.cat([visual_real, visual_imag], dim=-1)  # [bs, T, 2048]
        visual_out = self.visual_out_proj(visual_concat)  # [bs, T, 1024]

        # =====================================================================
        # Apply final layer norm
        # =====================================================================
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


class RotaryComplexTemporalAttentionModule(nn.Module):
    """
    Rotary Complex Temporal Attention Module.

    Inspired by RoPE (Rotary Position Embeddings), this module encodes
    temporal position as phase in the complex representation:

        x_t → x_t × e^(iθ_t) = x_t×cos(θ_t) + i×x_t×sin(θ_t)

    where θ_t = 2π × t / T (or learnable frequencies)

    Key insight:
    - Phase explicitly encodes "when" (temporal position)
    - Magnitude encodes "what" (content/features)
    - Complex attention Q @ K* computes relative temporal relationships
      through phase differences: e^(i(θ_q - θ_k))

    This is fundamentally different from arbitrary splits because:
    - Phase has explicit temporal meaning
    - Similar to how RoPE enables length generalization in LLMs
    - Complex arithmetic naturally computes relative positions

    Audio and visual are processed in separate streams, so final fusion
    with text (sim_at × sim_vt) remains meaningful.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        learnable_freq=True,  # Whether to learn frequency parameters
        num_freq_bands=None,  # If set, use multiple frequency bands (multi-scale)
        max_seq_len=10,  # Maximum sequence length
    ):
        super(RotaryComplexTemporalAttentionModule, self).__init__()

        self.feat_dim = feat_dim
        self.max_seq_len = max_seq_len
        self.learnable_freq = learnable_freq
        self.num_freq_bands = num_freq_bands if num_freq_bands else 1

        # Frequency parameters for rotary encoding
        if learnable_freq:
            # Learnable base frequency per feature dimension (or per band)
            # Initialize with geometric sequence like in RoPE
            if num_freq_bands and num_freq_bands > 1:
                # Multi-scale: different frequencies for different bands
                # Each band gets feat_dim // num_freq_bands dimensions
                self.band_dim = feat_dim // num_freq_bands
                # Base frequencies: 1, 2, 4, 8, ... (powers of 2)
                init_freqs = torch.tensor([2.0 ** i for i in range(num_freq_bands)])
                self.freq_bands = nn.Parameter(init_freqs)
            else:
                # Single frequency scale, but learnable per-dimension modulation
                # Initialize frequencies similar to RoPE: 1 / (10000^(2i/d))
                inv_freq = 1.0 / (10000 ** (torch.arange(0, feat_dim, 2).float() / feat_dim))
                self.inv_freq = nn.Parameter(inv_freq)
        else:
            # Fixed frequencies (standard rotary)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, feat_dim, 2).float() / feat_dim))
            self.register_buffer('inv_freq', inv_freq)

        # Create ComplexTransformerEncoder with FULL dimension
        block = ComplexTransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate self-attention blocks for audio and visual
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Output projection: we can either:
        # 1. Take magnitude: sqrt(real^2 + imag^2)
        # 2. Concat and project: [real, imag] -> Linear -> feat_dim
        # 3. Just take real (loses phase info at output)
        # Using concat + project to preserve information
        self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)

        # Output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

    def apply_rotary_encoding(self, x):
        """
        Apply rotary complex encoding to input features.

        Args:
            x: [bs, T, D] - input features

        Returns:
            real: [bs, T, D] - x * cos(θ)
            imag: [bs, T, D] - x * sin(θ)

        The encoding makes phase represent temporal position:
        - At t=0: θ ≈ 0, so real ≈ x, imag ≈ 0
        - At t=T/4: θ ≈ π/2, so real ≈ 0, imag ≈ x
        - At t=T/2: θ ≈ π, so real ≈ -x, imag ≈ 0
        - etc.

        This creates a smooth rotation through complex space as time progresses.
        """
        bs, T, D = x.shape
        device = x.device

        # Create position indices
        t = torch.arange(T, device=device).float()  # [T]

        if self.num_freq_bands and self.num_freq_bands > 1:
            # Multi-scale rotary encoding
            # Each band of features rotates at different frequency
            cos_parts = []
            sin_parts = []
            for i in range(self.num_freq_bands):
                freq = self.freq_bands[i] if self.learnable_freq else (2.0 ** i)
                theta = 2 * math.pi * freq * t / T  # [T]
                theta = theta.view(1, T, 1)  # [1, T, 1]

                start_idx = i * self.band_dim
                end_idx = start_idx + self.band_dim if i < self.num_freq_bands - 1 else D

                x_band = x[:, :, start_idx:end_idx]
                cos_parts.append(x_band * torch.cos(theta))
                sin_parts.append(x_band * torch.sin(theta))

            real = torch.cat(cos_parts, dim=-1)
            imag = torch.cat(sin_parts, dim=-1)
        else:
            # Standard rotary encoding (like RoPE)
            # Different frequencies for different dimension pairs
            # inv_freq: [D/2]
            # t: [T]
            # freqs: [T, D/2]
            freqs = torch.outer(t, self.inv_freq)  # [T, D/2]

            # Create full theta by repeating (for pairs of dimensions)
            # Each pair (2i, 2i+1) shares the same frequency
            theta = torch.cat([freqs, freqs], dim=-1)  # [T, D]
            theta = theta.unsqueeze(0)  # [1, T, D]

            # Apply rotation
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            real = x * cos_theta
            imag = x * sin_theta

        return real, imag

    def forward(self, a_tokens, v_tokens, text_embeddings=None):
        """
        Args:
            a_tokens: [bs, T, D] - audio features per frame (T=10, D=1024)
            v_tokens: [bs, T, D] - visual features per frame
            text_embeddings: [C, D] - IGNORED (kept for API compatibility)

        Returns:
            a_tokens: [bs, T, D] - refined audio features
            v_tokens: [bs, T, D] - refined visual features

        Process:
            1. Apply rotary complex encoding (phase = temporal position)
            2. Process through complex self-attention
            3. Concat real + imag, project back to D
        """
        # =====================================================================
        # AUDIO STREAM: Apply rotary encoding
        # =====================================================================
        audio_real, audio_imag = self.apply_rotary_encoding(a_tokens)

        # Pass through audio self-attention blocks
        for blk in self.a_sa_blocks:
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # Concatenate and project back to feat_dim
        audio_concat = torch.cat([audio_real, audio_imag], dim=-1)  # [bs, T, 2048]
        audio_out = self.audio_out_proj(audio_concat)  # [bs, T, 1024]

        # =====================================================================
        # VISUAL STREAM: Apply rotary encoding
        # =====================================================================
        visual_real, visual_imag = self.apply_rotary_encoding(v_tokens)

        # Pass through visual self-attention blocks
        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        # Concatenate and project back to feat_dim
        visual_concat = torch.cat([visual_real, visual_imag], dim=-1)  # [bs, T, 2048]
        visual_out = self.visual_out_proj(visual_concat)  # [bs, T, 1024]

        # =====================================================================
        # Apply final layer norm
        # =====================================================================
        audio_out = self.audio_norm(audio_out)
        visual_out = self.visual_norm(visual_out)

        # Check for NaN
        if torch.sum(audio_out.isnan()) > 0 or torch.sum(visual_out.isnan()) > 0:
            pdb.set_trace()

        return audio_out, visual_out


if __name__ == "__main__":
    temporal_scale = 10
    window_size = 4
    bs = 10
    n_heads = 1
    m = torch.zeros((temporal_scale, temporal_scale))
    w_len = window_size
    for j in range(temporal_scale):
        for k in range(w_len):
            m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
    # pdb.set_trace()
    m = m.repeat(bs, n_heads, 1, 1)

    a_tokens = torch.randn(2, 10, 1024)
    v_tokens = torch.randn(2, 10, 1024)
    tam = TemporalAttentionModule(
        sa_layer_num=2,
        xa_layer_num=2,
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=n_heads,
        dropout=0.1,
        use_adj_in_attn=True,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=True,
        win_size=4,
        norm_flag=None
    )
    new_a, new_v = tam(a_tokens, v_tokens)


    pdb.set_trace()


# ============================================================================
# Aligned Temporal Diff Complex Attention (For Paper)
# ============================================================================

class AlignedTemporalDiffComplexAttention(nn.Module):
    """
    Aligned Temporal Difference Complex Attention Module.

    Designed to justify complex-valued attention for temporal AV modeling:

    Key Design Principles:
    1. MEANINGFUL IMAGINARY: temporal diff (what's changing), not arbitrary split
    2. RESIDUAL CONNECTION: preserves original ImageBind feature alignment
    3. ALIGNMENT LOSS: explicit constraint to prevent feature drift
    4. FAIR PARAMETERS: same param count as real-valued baseline
    5. EXPLAINABLE: outputs phase/magnitude for visualization
    6. REAL-ONLY OUTPUT: uses only real component after attention (no concat+project)

    Hypothesis:
    - Real component = static features (appearance/sound content)
    - Imaginary component = temporal change (frame-to-frame dynamics)
    - Complex attention learns temporal relationships through phase
    - Phase = arctan(change/static) = "event-ness" of each frame
    - The imaginary (temporal diff) influences attention weights but
      the output is the refined real component only

    The alignment loss ensures features stay in ImageBind's well-structured
    cross-modal space while gaining temporal awareness.
    """

    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=1,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        residual_weight=0.5,  # Balance: higher = more original preservation
        alignment_loss_weight=0.1,  # Weight for alignment loss during training
    ):
        super(AlignedTemporalDiffComplexAttention, self).__init__()

        self.feat_dim = feat_dim
        self.residual_weight = residual_weight
        self.alignment_loss_weight = alignment_loss_weight

        # Complex transformer encoder - SAME dimension as input (fair params)
        block = ComplexTransformerEncoder(
            d_model=feat_dim,
            hid_dim=hid_dim,
            d_ff=d_ff,
            n_heads=head_num,
            dropout=dropout,
            use_adj_in_attn=use_adj_in_attn,
            gamma=gamma,
            bias=bias,
            use_mask_in_attn=use_mask_in_attn,
            win_size=win_size,
            norm_flag=norm_flag
        )

        # Separate streams for audio and visual (no parameter sharing)
        self.a_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])
        self.v_sa_blocks = nn.ModuleList([dcp(block) for _ in range(sa_layer_num)])

        # Layer norms for the refinement
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

        # Learnable residual gates (initialized to favor original features)
        # gate=0.5 means equal mix, gate>0.5 favors original
        self.audio_gate = nn.Parameter(torch.tensor(residual_weight))
        self.visual_gate = nn.Parameter(torch.tensor(residual_weight))

        # Output mode: use only real component (no concat+project)
        # Real = static features refined by complex attention
        # The imaginary (temporal diff) influences attention but output is real only

    def compute_temporal_diff(self, x):
        """
        Compute temporal differences: encodes "what's changing" between frames.

        For frame t: diff[t] = x[t] - x[t-1] (change TO this frame)
        For frame 0: diff[0] = x[1] - x[0] (forward diff, no previous)

        This has clear temporal meaning:
        - High diff = lots of change = likely event onset/offset
        - Low diff = stable = background or sustained event

        Args:
            x: [B, T, D] - input features
        Returns:
            diff: [B, T, D] - temporal differences
        """
        # Consecutive differences: x[t] - x[t-1] for t=1..T-1
        diff = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, D]

        # For frame 0, use forward difference (same as diff[0])
        first_diff = diff[:, 0:1, :]  # [B, 1, D]

        # Full sequence: [forward_diff_for_0, diff[0], diff[1], ..., diff[T-2]]
        full_diff = torch.cat([first_diff, diff], dim=1)  # [B, T, D]

        return full_diff

    def compute_phase_magnitude(self, real, imag):
        """
        Compute phase and magnitude for explainability.

        Phase = arctan(imag/real)
              = arctan(temporal_change / static_features)
              = "event-ness" - how much is changing relative to what's there

        Magnitude = sqrt(real² + imag²)
                  = overall "activity" level

        For paper visualization:
        - Phase should spike at event boundaries
        - Magnitude should be high during events
        """
        eps = 1e-8
        phase = torch.atan2(imag, real + eps)  # [B, T, D]
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + eps)  # [B, T, D]
        return phase, magnitude

    def compute_alignment_loss(self, original, refined):
        """
        Compute alignment loss to keep features in ImageBind space.

        Uses cosine similarity: features should point in same direction.
        This allows magnitude changes (attention can amplify/suppress)
        but preserves semantic direction.

        Args:
            original: [B, T, D] - original ImageBind features
            refined: [B, T, D] - refined features after attention

        Returns:
            loss: scalar - alignment loss (0 = perfectly aligned)
        """
        # Flatten to [B*T, D] for per-frame alignment
        orig_flat = original.reshape(-1, original.size(-1))
        ref_flat = refined.reshape(-1, refined.size(-1))

        # Cosine similarity: 1 = same direction, -1 = opposite
        cos_sim = F.cosine_similarity(orig_flat, ref_flat, dim=-1)

        # Loss: want cos_sim = 1, so loss = 1 - cos_sim
        alignment_loss = (1 - cos_sim).mean()

        return alignment_loss

    def forward(self, a_tokens, v_tokens, return_losses=False, return_explainability=False):
        """
        Forward pass with optional alignment loss and explainability outputs.

        Args:
            a_tokens: [B, T, D] - audio features from ImageBind
            v_tokens: [B, T, D] - visual features from ImageBind
            return_losses: if True, return alignment loss for training
            return_explainability: if True, return phase/magnitude for analysis

        Returns:
            a_out: [B, T, D] - temporally-refined audio features
            v_out: [B, T, D] - temporally-refined visual features
            (optional) losses: dict with 'alignment_loss'
            (optional) explain: dict with phase, magnitude, etc.
        """
        # Store originals for residual and alignment
        a_original = a_tokens
        v_original = v_tokens

        # =====================================================================
        # AUDIO STREAM
        # =====================================================================
        # Real = static features, Imag = temporal change (meaningful!)
        audio_real = a_tokens
        audio_imag = self.compute_temporal_diff(a_tokens)

        # Store pre-attention phase for explainability
        if return_explainability:
            audio_phase_pre, audio_mag_pre = self.compute_phase_magnitude(audio_real, audio_imag)

        # Complex self-attention: learns temporal relationships via phase
        for blk in self.a_sa_blocks:
            audio_real, audio_imag = blk(audio_real, audio_imag)

        # Store post-attention phase for explainability
        if return_explainability:
            audio_phase_post, audio_mag_post = self.compute_phase_magnitude(audio_real, audio_imag)

        # Gated residual: blend original with refined
        # gate > 0.5 favors original (more alignment preservation)
        gate_a = torch.sigmoid(self.audio_gate)
        # Use only real output (imaginary influences attention but not final output)
        audio_refinement = self.audio_norm(audio_real)
        a_out = gate_a * a_original + (1 - gate_a) * audio_refinement

        # =====================================================================
        # VISUAL STREAM
        # =====================================================================
        visual_real = v_tokens
        visual_imag = self.compute_temporal_diff(v_tokens)

        if return_explainability:
            visual_phase_pre, visual_mag_pre = self.compute_phase_magnitude(visual_real, visual_imag)

        for blk in self.v_sa_blocks:
            visual_real, visual_imag = blk(visual_real, visual_imag)

        if return_explainability:
            visual_phase_post, visual_mag_post = self.compute_phase_magnitude(visual_real, visual_imag)

        gate_v = torch.sigmoid(self.visual_gate)
        # Use only real output (imaginary influences attention but not final output)
        visual_refinement = self.visual_norm(visual_real)
        v_out = gate_v * v_original + (1 - gate_v) * visual_refinement

        # =====================================================================
        # Prepare outputs
        # =====================================================================

        # Check for NaN
        if torch.sum(a_out.isnan()) > 0 or torch.sum(v_out.isnan()) > 0:
            pdb.set_trace()

        # Base output
        outputs = (a_out, v_out)

        # Optional: alignment loss for training
        if return_losses:
            align_loss_a = self.compute_alignment_loss(a_original, a_out)
            align_loss_v = self.compute_alignment_loss(v_original, v_out)
            total_align_loss = (align_loss_a + align_loss_v) / 2

            losses = {
                'alignment_loss': total_align_loss,
                'audio_alignment': align_loss_a,
                'visual_alignment': align_loss_v,
                'audio_gate': gate_a.item(),
                'visual_gate': gate_v.item(),
            }
            outputs = outputs + (losses,)

        # Optional: explainability outputs for visualization
        if return_explainability:
            explain = {
                # Raw per-feature phases for proper MPC computation [B, T, D]
                'audio_phase_raw': audio_phase_post,
                'visual_phase_raw': visual_phase_post,
                # Phase before/after attention (for visualization) [B, T]
                'audio_phase_pre': audio_phase_pre.mean(dim=-1),   # [B, T] avg over features
                'audio_phase_post': audio_phase_post.mean(dim=-1),
                'visual_phase_pre': visual_phase_pre.mean(dim=-1),
                'visual_phase_post': visual_phase_post.mean(dim=-1),
                # Magnitude before/after
                'audio_mag_pre': audio_mag_pre.mean(dim=-1),
                'audio_mag_post': audio_mag_post.mean(dim=-1),
                'visual_mag_pre': visual_mag_pre.mean(dim=-1),
                'visual_mag_post': visual_mag_post.mean(dim=-1),
                # Temporal diffs (what the network sees as "change")
                'audio_temporal_diff': self.compute_temporal_diff(a_original).mean(dim=-1),
                'visual_temporal_diff': self.compute_temporal_diff(v_original).mean(dim=-1),
                # Gates
                'audio_gate': gate_a.item(),
                'visual_gate': gate_v.item(),
            }
            outputs = outputs + (explain,)

        return outputs


# ============================================================================
# Phase-Aware Complex Attention with Hilbert Projection and ModReLU
# ============================================================================

class HilbertProjection(nn.Module):
    """
    Complex Lift: Real features -> Complex representation

    Modes (imag_mode):
    - 'temporal_diff': Imag = temporal differences (x[t+1] - x[t]) - original Hilbert
    - 'zero': Imag = zeros (pure TAPA mode)
    - 'learned': Imag = learned linear projection of real (recommended for TAPA)

    - Real part: semantic content (the features themselves)
    - Imag part: depends on mode
    """
    def __init__(self, feat_dim=1024, use_temporal_bias=True, use_temporal_diff=True, imag_mode=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_temporal_bias = use_temporal_bias

        # Determine imag_mode from parameters
        # For backward compatibility: use_temporal_diff=True means 'temporal_diff', False means check imag_mode
        if imag_mode is not None:
            self.imag_mode = imag_mode
        elif use_temporal_diff:
            self.imag_mode = 'temporal_diff'
        else:
            self.imag_mode = 'zero'

        # Fixed scale for temporal diff mode
        self.register_buffer('imag_scale', torch.tensor(0.1))

        # Learned projection for 'learned' mode
        if self.imag_mode == 'learned':
            self.imag_proj = nn.Linear(feat_dim, feat_dim)
            # Initialize with small weights to not dominate initially
            nn.init.xavier_uniform_(self.imag_proj.weight, gain=0.1)
            nn.init.zeros_(self.imag_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] real-valued features from ImageBind
        Returns:
            real: [B, T, D] - semantic component
            imag: [B, T, D] - depends on imag_mode
        """
        B, T, D = x.shape

        # Real part: just the input features (semantic content)
        real = x

        if self.imag_mode == 'temporal_diff':
            # Imag part: temporal difference (captures dynamics)
            # Forward difference: x[t+1] - x[t], with zero padding at the end
            diff = torch.zeros_like(x)
            diff[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
            imag = self.imag_scale * diff
        elif self.imag_mode == 'learned':
            # Learned projection: network decides what goes in imaginary
            imag = self.imag_proj(x)
        else:  # 'zero'
            # Pure mode: let TAPA's complex projections handle imaginary part
            imag = torch.zeros_like(x)

        return real, imag


class ModReLU(nn.Module):
    """
    Modulus ReLU: ReLU(|z| + b) * e^(iθ)

    From Deep Complex Networks (Trabelsi et al., ICLR 2018)

    Key property: Phase is EXACTLY preserved.
    - Applies ReLU only to magnitude
    - Phase passes through unchanged
    - Learnable bias b creates a dead zone of radius |b| around origin

    This is CRITICAL for phase-based synchronization because standard
    ReLU (CReLU) on real/imag independently scrambles the phase relationship.

    Note: bias_init should be slightly negative to create a small dead zone
    initially, allowing the network to learn which magnitudes matter.
    """
    def __init__(self, num_features, bias_init=-0.1):
        super().__init__()
        # Learnable bias per feature dimension
        self.bias = nn.Parameter(torch.full((num_features,), bias_init))

    def forward(self, real, imag):
        """
        Args:
            real, imag: [B, T, D] complex components
        Returns:
            real_out, imag_out: [B, T, D] with preserved phase
        """
        # Compute magnitude: |z| = sqrt(real² + imag²)
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)

        # Apply ReLU to (magnitude + bias)
        # bias < 0 creates dead zone (small magnitudes zeroed)
        # bias > 0 allows everything through
        new_magnitude = F.relu(magnitude + self.bias)

        # NUMERICAL STABILITY FIX: Scale factor to preserve phase without atan2
        # Instead of phase = atan2(imag, real) then cos/sin,
        # use direct scaling: new_z = (new_mag / old_mag) * z
        # This avoids atan2 instabilities when real/imag are both small
        scale = new_magnitude / (magnitude + 1e-8)

        real_out = scale * real
        imag_out = scale * imag

        return real_out, imag_out


class PhaseAwareComplexFFN(nn.Module):
    """
    Complex-valued Feed-Forward Network with ModReLU activation.

    Replaces ComplexPositionwiseFeedForward which uses CReLU (phase-destroying).

    Structure: Linear -> ModReLU (phase-preserving) -> Dropout -> Linear
    """
    def __init__(self, d_model, d_ff, dropout=0.1, modrelu_bias_init=-0.1):
        super().__init__()
        self.w_1 = ComplexLinear(d_model, d_ff)
        self.w_2 = ComplexLinear(d_ff, d_model)
        self.activation = ModReLU(d_ff, bias_init=modrelu_bias_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, real, imag):
        """
        Args:
            real, imag: [B, T, D] complex components
        Returns:
            real_out, imag_out: [B, T, D]
        """
        # First linear projection (complex multiplication)
        real, imag = self.w_1(real, imag)

        # NUMERICAL STABILITY FIX: Clamp values to prevent explosion
        real = torch.clamp(real, min=-100.0, max=100.0)
        imag = torch.clamp(imag, min=-100.0, max=100.0)

        # ModReLU: phase-preserving activation
        real, imag = self.activation(real, imag)

        # NUMERICAL STABILITY FIX: Apply dropout directly to real/imag components
        # instead of magnitude-phase decomposition which can cause 0/0 issues
        real = self.dropout(real)
        imag = self.dropout(imag)

        # Second linear projection
        real, imag = self.w_2(real, imag)

        # NUMERICAL STABILITY FIX: Final clamp to prevent unbounded growth
        real = torch.clamp(real, min=-100.0, max=100.0)
        imag = torch.clamp(imag, min=-100.0, max=100.0)

        return real, imag


class PhaseAwareComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex values that preserves phase relationships.

    Option 1 (default): Normalize magnitude, preserve phase
    Option 2: Separate normalization (like ComplexLayerNorm but with phase tracking)
    """
    def __init__(self, d_model, mode='magnitude'):
        super().__init__()
        self.mode = mode
        if mode == 'magnitude':
            # Normalize magnitude, preserve phase
            self.norm = nn.LayerNorm(d_model)
        else:
            # Separate normalization (traditional)
            self.norm_real = nn.LayerNorm(d_model)
            self.norm_imag = nn.LayerNorm(d_model)

    def forward(self, real, imag):
        if self.mode == 'magnitude':
            # Compute magnitude
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)

            # Normalize magnitude only
            magnitude_norm = self.norm(magnitude)

            # CRITICAL: LayerNorm can produce negative values, but magnitude must be non-negative
            # Clamp to ensure valid scaling
            magnitude_norm = torch.clamp(magnitude_norm, min=1e-8)

            # NUMERICAL STABILITY FIX: Use direct scaling instead of phase reconstruction
            # This avoids atan2 instabilities when real/imag are both small
            scale = magnitude_norm / (magnitude + 1e-8)

            real_out = scale * real
            imag_out = scale * imag
            return real_out, imag_out
        else:
            return self.norm_real(real), self.norm_imag(imag)


class PhaseAwareComplexMultiHeadAttention(nn.Module):
    """
    Complex-valued Multi-Head Attention with phase-aware attention scoring.

    Key difference from ComplexMultiHeadAttention:
    - Uses both real AND imaginary parts of Q @ K^H for attention
    - The imaginary part captures phase relationships (temporal synchronization)

    Complex attention: Q @ K^H where K^H is conjugate transpose
    (Q_r + i*Q_i) @ (K_r - i*K_i)^T =
        Real: (Q_r @ K_r^T + Q_i @ K_i^T)  <- magnitude correlation
        Imag: (Q_i @ K_r^T - Q_r @ K_i^T)  <- phase relationship

    We use: softmax(Real / sqrt(d)) for attention weights
    But the phase info from Imag influences V through complex arithmetic.
    """
    def __init__(self, d_model, dim_k, dim_v, n_heads, modrelu_bias_init=-0.1):
        super().__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        # Complex Q, K, V projections
        self.q = ComplexLinear(d_model, dim_k)
        self.k = ComplexLinear(d_model, dim_k)
        self.v = ComplexLinear(d_model, dim_v)
        self.o = ComplexLinear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k // n_heads)
        self.act = nn.Softmax(dim=-1)

    def forward(self, query_real, query_imag, key_real, key_imag, value_real, value_imag, mask=None, adj=None):
        """
        Args:
            query_real, query_imag: [bs, T, d_model]
            key_real, key_imag: [bs, T, d_model]
            value_real, value_imag: [bs, T, d_model]
        Returns:
            out_real, out_imag: [bs, T, d_model]
        """
        bs = query_real.shape[0]
        seq_len = query_real.shape[1]
        head_dim = self.dim_k // self.n_heads

        # Complex Q, K, V projections
        Q_real, Q_imag = self.q(query_real, query_imag)
        K_real, K_imag = self.k(key_real, key_imag)
        V_real, V_imag = self.v(value_real, value_imag)

        # Reshape for multi-head: [bs, seq, dim] -> [bs, heads, seq, dim/heads]
        Q_real = Q_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        Q_imag = Q_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        K_real = K_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        K_imag = K_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        V_real = V_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        V_imag = V_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # =====================================================================
        # PROPER COMPLEX ATTENTION (per "Building Blocks for Complex Transformer")
        # =====================================================================
        # Complex attention scores: Q @ K^H (conjugate transpose)
        # Real part: Q_r @ K_r^T + Q_i @ K_i^T (correlation in magnitude)
        # Imag part: Q_i @ K_r^T - Q_r @ K_i^T (phase difference)
        attn_real = torch.matmul(Q_real, K_real.transpose(-2, -1)) + torch.matmul(Q_imag, K_imag.transpose(-2, -1))
        attn_imag = torch.matmul(Q_imag, K_real.transpose(-2, -1)) - torch.matmul(Q_real, K_imag.transpose(-2, -1))

        # Scale both parts
        attn_real = attn_real * self.norm_fact
        attn_imag = attn_imag * self.norm_fact

        # Compute magnitude and phase of complex attention scores
        attn_magnitude = torch.sqrt(attn_real ** 2 + attn_imag ** 2 + 1e-8)
        attn_phase = torch.atan2(attn_imag, attn_real + 1e-8)

        if adj is not None:
            attn_magnitude = attn_magnitude + adj

        if mask is not None:
            attn_magnitude = attn_magnitude.masked_fill_(mask.data.eq(0), -1e9)

        # NUMERICAL STABILITY FIX: Clamp magnitude before softmax
        attn_magnitude = torch.clamp(attn_magnitude, min=-100.0, max=100.0)

        # Softmax on MAGNITUDE (gives valid probability distribution)
        attn_weights_mag = self.act(attn_magnitude)  # [bs, heads, seq, seq]

        # Reconstruct complex attention weights: magnitude * e^(i*phase)
        attn_weights_real = attn_weights_mag * torch.cos(attn_phase)
        attn_weights_imag = attn_weights_mag * torch.sin(attn_phase)

        # COMPLEX MULTIPLICATION: (attn_r + i*attn_i) @ (V_r + i*V_i)
        out_real = torch.matmul(attn_weights_real, V_real) - torch.matmul(attn_weights_imag, V_imag)
        out_imag = torch.matmul(attn_weights_real, V_imag) + torch.matmul(attn_weights_imag, V_real)

        # Reshape back: [bs, heads, seq, dim/heads] -> [bs, seq, dim]
        out_real = out_real.transpose(1, 2).reshape(bs, seq_len, -1)
        out_imag = out_imag.transpose(1, 2).reshape(bs, seq_len, -1)

        # Output projection
        out_real, out_imag = self.o(out_real, out_imag)

        return out_real, out_imag


class TAPAComplexMultiHeadAttention(nn.Module):
    """
    Token-Aware Phase Attention (TAPA) integrated with Complex-valued Attention.

    From paper: "Positional Encoding via Token-Aware Phase Attention"
    (https://arxiv.org/abs/2509.12635)

    TAPA Formula:
        Attn(q,k) = (q_A^T · k_A)/√(θD) · cos(2π|m-n|^α · q_P^T · k_P/√((1-θ)D))
                    ─────────────────────   ──────────────────────────────────────
                    Amplitude (content)     Phase (position-modulated)

    Integration with Complex Numbers:
        - Complex Q, K naturally have magnitude (amplitude) and phase
        - We split Q, K dimensions: first θD for amplitude, rest for phase
        - Amplitude attention: standard dot product on amplitude dimensions
        - Phase attention: cosine-modulated by relative distance |m-n|

    Key Benefits:
        - Eliminates distance-dependent bias (unlike RoPE)
        - Learns position-aware interactions through phase
        - No additional parameters vs vanilla transformer
        - Natural fit with complex representation
    """

    def __init__(
        self,
        d_model,
        dim_k,
        dim_v,
        n_heads,
        theta=0.1,        # Split ratio: θD for amplitude, (1-θ)D for phase
        alpha=0.5,        # Distance scaling exponent
        modrelu_bias_init=-0.1,
    ):
        super().__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads
        self.theta = theta
        self.alpha = alpha

        # Complex Q, K, V projections
        self.q = ComplexLinear(d_model, dim_k)
        self.k = ComplexLinear(d_model, dim_k)
        self.v = ComplexLinear(d_model, dim_v)
        self.o = ComplexLinear(dim_v, d_model)

        # Compute dimension splits per head
        head_dim = dim_k // n_heads
        self.amp_dim = max(1, int(head_dim * theta))  # Amplitude dimensions
        self.phase_dim = head_dim - self.amp_dim       # Phase dimensions

        # Scaling factors
        self.amp_scale = 1 / math.sqrt(self.amp_dim)
        self.phase_scale = 1 / math.sqrt(self.phase_dim) if self.phase_dim > 0 else 1.0

        self.act = nn.Softmax(dim=-1)

        # Learnable phase scaling (optional enhancement)
        self.phase_weight = nn.Parameter(torch.ones(1) * 2 * math.pi)

    def _compute_distance_matrix(self, seq_len, device):
        """
        Compute |m - n| distance matrix for positions.

        Returns:
            dist: [seq_len, seq_len] where dist[m, n] = |m - n|
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # |m - n| for all pairs
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        return dist

    def forward(self, query_real, query_imag, key_real, key_imag, value_real, value_imag, mask=None, adj=None):
        """
        TAPA-enhanced complex attention.

        Attention = softmax(Amplitude_score · cos(Phase_score · |m-n|^α))

        Args:
            query_real, query_imag: [bs, T, d_model]
            key_real, key_imag: [bs, T, d_model]
            value_real, value_imag: [bs, T, d_model]
        Returns:
            out_real, out_imag: [bs, T, d_model]
        """
        bs = query_real.shape[0]
        seq_len = query_real.shape[1]
        head_dim = self.dim_k // self.n_heads

        # Complex Q, K, V projections
        Q_real, Q_imag = self.q(query_real, query_imag)
        K_real, K_imag = self.k(key_real, key_imag)
        V_real, V_imag = self.v(value_real, value_imag)

        # Reshape for multi-head: [bs, seq, dim] -> [bs, heads, seq, dim/heads]
        Q_real = Q_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        Q_imag = Q_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        K_real = K_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        K_imag = K_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        V_real = V_real.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)
        V_imag = V_imag.view(bs, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # =====================================================================
        # TAPA: Split into Amplitude and Phase components
        # =====================================================================

        # Split Q and K into amplitude and phase parts along feature dimension
        # Amplitude: first amp_dim features (content similarity)
        # Phase: remaining phase_dim features (position encoding)

        Q_amp_real = Q_real[..., :self.amp_dim]  # [bs, heads, seq, amp_dim]
        Q_amp_imag = Q_imag[..., :self.amp_dim]
        K_amp_real = K_real[..., :self.amp_dim]
        K_amp_imag = K_imag[..., :self.amp_dim]

        Q_phase_real = Q_real[..., self.amp_dim:]  # [bs, heads, seq, phase_dim]
        Q_phase_imag = Q_imag[..., self.amp_dim:]
        K_phase_real = K_real[..., self.amp_dim:]
        K_phase_imag = K_imag[..., self.amp_dim:]

        # =====================================================================
        # Amplitude Attention (content-based, like standard attention)
        # =====================================================================
        # Complex dot product for amplitude: Re(Q_amp @ K_amp^H)
        amp_score = (
            torch.matmul(Q_amp_real, K_amp_real.transpose(-2, -1)) +
            torch.matmul(Q_amp_imag, K_amp_imag.transpose(-2, -1))
        ) * self.amp_scale  # [bs, heads, seq, seq]

        # =====================================================================
        # Phase Attention (position-modulated, TAPA's key contribution)
        # =====================================================================
        # Complex dot product for phase: Re(Q_phase @ K_phase^H)
        phase_interaction = (
            torch.matmul(Q_phase_real, K_phase_real.transpose(-2, -1)) +
            torch.matmul(Q_phase_imag, K_phase_imag.transpose(-2, -1))
        ) * self.phase_scale  # [bs, heads, seq, seq]

        # Distance matrix: |m - n|^α
        dist = self._compute_distance_matrix(seq_len, query_real.device)  # [seq, seq]
        dist_scaled = torch.pow(dist + 1e-8, self.alpha)  # [seq, seq], add eps to avoid 0^α issues

        # TAPA phase modulation: cos(2π * |m-n|^α * phase_interaction)
        # NUMERICAL STABILITY FIX: Clamp the phase argument to prevent cos() instability
        # Large arguments to cos() cause numerical issues and meaningless gradients
        phase_arg = self.phase_weight * dist_scaled.unsqueeze(0).unsqueeze(0) * phase_interaction
        phase_arg = torch.clamp(phase_arg, min=-50.0, max=50.0)  # Prevent exploding arguments
        phase_score = torch.cos(phase_arg)
        # [bs, heads, seq, seq]

        # =====================================================================
        # Combined TAPA Attention
        # =====================================================================
        # NUMERICAL STABILITY FIX: Use additive combination instead of multiplicative
        # Multiplicative can drive scores to zero when phase_score ≈ 0, causing softmax NaN
        # Additive: amp provides base similarity, phase modulates it
        attn_scores = amp_score + 0.5 * (phase_score - 1.0)  # phase_score in [-1,1], so this adds [-1, 0]

        if adj is not None:
            attn_scores = attn_scores + adj

        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask.data.eq(0), -1e9)

        # NUMERICAL STABILITY FIX: Ensure attention scores don't cause softmax issues
        attn_scores = torch.clamp(attn_scores, min=-100.0, max=100.0)
        attn_weights = self.act(attn_scores)  # [bs, heads, seq, seq]

        # Apply attention to complex values
        out_real = torch.matmul(attn_weights, V_real)
        out_imag = torch.matmul(attn_weights, V_imag)

        # Reshape back: [bs, heads, seq, dim/heads] -> [bs, seq, dim]
        out_real = out_real.transpose(1, 2).reshape(bs, seq_len, -1)
        out_imag = out_imag.transpose(1, 2).reshape(bs, seq_len, -1)

        # Output projection
        out_real, out_imag = self.o(out_real, out_imag)

        return out_real, out_imag


# =============================================================================
# Pure Real-Valued TAPA (Original Paper Implementation)
# =============================================================================

class TAPAMultiHeadAttention(nn.Module):
    """
    Pure Real-Valued Token-Aware Phase Attention (TAPA).

    This is the ORIGINAL TAPA from the paper "Positional Encoding via
    Token-Aware Phase Attention" (arXiv:2509.12635).

    Unlike TAPAComplexMultiHeadAttention, this operates entirely on real numbers.
    No Hilbert projection or complex arithmetic required.

    TAPA Formula:
        Attn(q,k) = (q_A @ k_A^T)/√(θD) · cos(2π|m-n|^α · q_P @ k_P^T/√((1-θ)D))
                    ─────────────────────   ───────────────────────────────────────
                    Amplitude (content)      Phase (position-modulated)

    How dimension splitting works:
        Given head_dim = D/n_heads, for each head:
        - Amplitude dims: first θ·head_dim features  → content similarity
        - Phase dims: remaining (1-θ)·head_dim features → positional encoding

        Example with head_dim=64, θ=0.1:
        - Amplitude: dims 0-5 (6 dims)   → "what" information
        - Phase: dims 6-63 (58 dims)     → "where" information

    Key Benefits over standard attention:
        1. No separate positional embeddings needed
        2. Learns which positional patterns matter (via phase dims)
        3. Distance-aware: |m-n| explicitly encodes relative position
        4. No extra parameters vs vanilla transformer
    """

    def __init__(
        self,
        d_model,
        dim_k,
        dim_v,
        n_heads,
        theta=0.1,      # Split ratio: θD for amplitude, (1-θ)D for phase
        alpha=0.5,      # Distance scaling exponent |m-n|^α
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.theta = theta
        self.alpha = alpha

        # Standard Q, K, V projections (real-valued)
        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        # Compute dimension splits per head
        head_dim = dim_k // n_heads
        self.head_dim = head_dim
        self.amp_dim = max(1, int(head_dim * theta))    # Amplitude dimensions
        self.phase_dim = head_dim - self.amp_dim        # Phase dimensions

        # Scaling factors
        self.amp_scale = 1.0 / math.sqrt(self.amp_dim)
        self.phase_scale = 1.0 / math.sqrt(self.phase_dim) if self.phase_dim > 0 else 1.0

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # Learnable phase weight (2π in original paper, but learnable can help)
        self.phase_weight = nn.Parameter(torch.tensor(2.0 * math.pi))

        print(f"    [TAPAMultiHeadAttention] head_dim={head_dim}, "
              f"amp_dim={self.amp_dim} ({theta*100:.0f}%), "
              f"phase_dim={self.phase_dim} ({(1-theta)*100:.0f}%)")

    def _compute_distance_matrix(self, seq_len, device):
        """
        Compute |m - n|^α distance matrix.

        For seq_len=4:
            positions = [0, 1, 2, 3]
            dist[i,j] = |i - j|^α

            With α=0.5 (sqrt):
            [[0.00, 1.00, 1.41, 1.73],
             [1.00, 0.00, 1.00, 1.41],
             [1.41, 1.00, 0.00, 1.00],
             [1.73, 1.41, 1.00, 0.00]]
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # |m - n| for all position pairs
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        # Apply exponent: |m - n|^α
        dist_scaled = torch.pow(dist + 1e-8, self.alpha)  # eps to avoid 0^α issues
        return dist_scaled

    def forward(self, query, key, value, mask=None, adj=None, return_attn_weights=False):
        """
        Pure real-valued TAPA forward pass.

        Args:
            query: [B, T, D] - query features (real)
            key: [B, T, D] - key features (real)
            value: [B, T, D] - value features (real)
            mask: Optional attention mask
            adj: Optional adjacency bias
            return_attn_weights: Whether to return attention weights

        Returns:
            output: [B, T, D] - attended features (real)
            attn_weights (optional): [B, H, T, T] - attention weights
        """
        B, T, _ = query.shape

        # =====================================================================
        # Step 1: Linear projections Q, K, V
        # =====================================================================
        Q = self.q(query)  # [B, T, dim_k]
        K = self.k(key)    # [B, T, dim_k]
        V = self.v(value)  # [B, T, dim_v]

        # Reshape for multi-head: [B, T, dim] -> [B, H, T, dim/H]
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.dim_v // self.n_heads).transpose(1, 2)

        # =====================================================================
        # Step 2: Split Q and K into Amplitude and Phase parts
        # =====================================================================
        # This is the key TAPA operation - dimension splitting
        #
        # Q = [q_0, q_1, ..., q_{D-1}]
        #      └── amp_dim ──┘└── phase_dim ──┘
        #
        Q_amp = Q[..., :self.amp_dim]      # [B, H, T, amp_dim] - content features
        Q_phase = Q[..., self.amp_dim:]    # [B, H, T, phase_dim] - position features

        K_amp = K[..., :self.amp_dim]      # [B, H, T, amp_dim]
        K_phase = K[..., self.amp_dim:]    # [B, H, T, phase_dim]

        # =====================================================================
        # Step 3: Amplitude Attention (content-based, like standard attention)
        # =====================================================================
        # amp_score = (Q_amp @ K_amp^T) / √(amp_dim)
        # This captures semantic/content similarity
        amp_score = torch.matmul(Q_amp, K_amp.transpose(-2, -1)) * self.amp_scale
        # Shape: [B, H, T, T]

        # =====================================================================
        # Step 4: Phase Attention (position-modulated, TAPA's key contribution)
        # =====================================================================
        # phase_interaction = (Q_phase @ K_phase^T) / √(phase_dim)
        # This learns which positional relationships matter
        phase_interaction = torch.matmul(Q_phase, K_phase.transpose(-2, -1)) * self.phase_scale
        # Shape: [B, H, T, T]

        # Distance matrix: |m - n|^α
        dist = self._compute_distance_matrix(T, query.device)  # [T, T]

        # TAPA phase modulation: cos(2π · |m-n|^α · phase_interaction)
        # - dist encodes "how far apart" positions are
        # - phase_interaction learns "what distance patterns matter"
        # - cosine creates periodic positional sensitivity
        phase_arg = self.phase_weight * dist.unsqueeze(0).unsqueeze(0) * phase_interaction

        # Numerical stability: clamp to prevent cos() issues
        phase_arg = torch.clamp(phase_arg, min=-50.0, max=50.0)
        phase_score = torch.cos(phase_arg)  # [B, H, T, T], values in [-1, 1]

        # =====================================================================
        # Step 5: Combine Amplitude and Phase scores
        # =====================================================================
        # Original paper: multiplicative combination
        # attn_scores = amp_score * phase_score
        #
        # However, multiplicative can cause issues when phase_score ≈ 0
        # We use additive for numerical stability:
        # attn_scores = amp_score + λ * (phase_score - 1)
        #
        # This way:
        # - When phase_score = 1 (same position or aligned): no penalty
        # - When phase_score = -1 (misaligned): penalty of -2λ
        attn_scores = amp_score + 0.5 * (phase_score - 1.0)

        # Optional adjacency bias
        if adj is not None:
            attn_scores = attn_scores + adj

        # Optional mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Numerical stability before softmax
        attn_scores = torch.clamp(attn_scores, min=-100.0, max=100.0)

        # =====================================================================
        # Step 6: Softmax and apply to values
        # =====================================================================
        attn_weights = self.softmax(attn_scores)  # [B, H, T, T]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, H, T, dim_v/H]

        # Reshape back: [B, H, T, dim_v/H] -> [B, T, dim_v]
        output = output.transpose(1, 2).reshape(B, T, -1)

        # Output projection
        output = self.o(output)  # [B, T, d_model]

        if return_attn_weights:
            return output, attn_weights
        return output


class TAPATransformerBlock(nn.Module):
    """
    Transformer block using pure real-valued TAPA attention.

    Drop-in replacement for standard transformer encoder block.
    No complex arithmetic, no Hilbert projection needed.
    """

    def __init__(
        self,
        d_model=1024,
        dim_k=256,
        dim_v=256,
        d_ff=512,
        n_heads=4,
        dropout=0.1,
        theta=0.1,
        alpha=0.5,
    ):
        super().__init__()

        self.self_attn = TAPAMultiHeadAttention(
            d_model=d_model,
            dim_k=dim_k,
            dim_v=dim_v,
            n_heads=n_heads,
            theta=theta,
            alpha=alpha,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, D] input features (real-valued)
            mask: Optional attention mask
        Returns:
            x: [B, T, D] output features (real-valued)
        """
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TAPATemporalAttention(nn.Module):
    """
    Complete TAPA-based temporal attention module for audio-visual features.

    This is the pure real-valued version that can replace existing temporal
    attention modules without requiring complex-valued inputs.

    Usage:
        # In config.py, set:
        cfg.TA.USE_PURE_TAPA = True
        cfg.TA.TAPA_THETA = 0.1
        cfg.TA.TAPA_ALPHA = 0.5

        # In model:
        self.temporal_layer = TAPATemporalAttention(
            d_model=1024,
            theta=0.1,
            alpha=0.5,
        )

        # Forward:
        audio_out, visual_out = self.temporal_layer(audio_feat, visual_feat)
    """

    def __init__(
        self,
        d_model=1024,
        hid_dim=256,
        d_ff=512,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
        theta=0.1,
        alpha=0.5,
        use_gated_residual=True,
    ):
        super().__init__()

        self.use_gated_residual = use_gated_residual

        # Separate TAPA blocks for audio and visual (can share weights if desired)
        self.audio_layers = nn.ModuleList([
            TAPATransformerBlock(
                d_model=d_model,
                dim_k=hid_dim,
                dim_v=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                theta=theta,
                alpha=alpha,
            )
            for _ in range(n_layers)
        ])

        self.visual_layers = nn.ModuleList([
            TAPATransformerBlock(
                d_model=d_model,
                dim_k=hid_dim,
                dim_v=hid_dim,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                theta=theta,
                alpha=alpha,
            )
            for _ in range(n_layers)
        ])

        # Optional gated residual to preserve original features
        if use_gated_residual:
            self.audio_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.visual_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

        print(f"[TAPATemporalAttention] Pure real-valued TAPA")
        print(f"    theta={theta}, alpha={alpha}, n_layers={n_layers}")
        print(f"    gated_residual={use_gated_residual}")

    def forward(self, audio_feat, visual_feat, mask=None):
        """
        Args:
            audio_feat: [B, T, D] audio features from ImageBind
            visual_feat: [B, T, D] visual features from ImageBind
            mask: Optional attention mask

        Returns:
            audio_out: [B, T, D] temporally-attended audio features
            visual_out: [B, T, D] temporally-attended visual features
        """
        # Store original for residual
        audio_orig = audio_feat
        visual_orig = visual_feat

        # Apply TAPA layers
        for layer in self.audio_layers:
            audio_feat = layer(audio_feat, mask=mask)

        for layer in self.visual_layers:
            visual_feat = layer(visual_feat, mask=mask)

        # Gated residual connection
        if self.use_gated_residual:
            audio_gate = self.audio_gate(torch.cat([audio_orig, audio_feat], dim=-1))
            audio_out = audio_gate * audio_feat + (1 - audio_gate) * audio_orig

            visual_gate = self.visual_gate(torch.cat([visual_orig, visual_feat], dim=-1))
            visual_out = visual_gate * visual_feat + (1 - visual_gate) * visual_orig
        else:
            audio_out = audio_feat
            visual_out = visual_feat

        return audio_out, visual_out


class PhaseAwareComplexTransformerBlock(nn.Module):
    """
    Single transformer block with phase-aware complex operations.

    Structure:
    1. Complex Self-Attention (with phase-aware scoring, optionally TAPA-enhanced)
    2. Add & Norm (phase-preserving)
    3. FFN with ModReLU (phase-preserving activation)
    4. Add & Norm (phase-preserving)

    TAPA Integration (when use_tapa=True):
        Uses Token-Aware Phase Attention from arXiv:2509.12635
        - Splits Q, K into amplitude and phase components
        - Amplitude: content similarity (standard dot product)
        - Phase: position-modulated by |m-n|^α with learned interaction
    """
    def __init__(
        self,
        d_model=1024,
        hid_dim=256,
        d_ff=512,
        n_heads=4,
        dropout=0.1,
        modrelu_bias_init=-0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        # TAPA parameters
        use_tapa=False,
        tapa_theta=0.1,
        tapa_alpha=0.5,
    ):
        super().__init__()

        # Choose attention type based on TAPA flag
        if use_tapa:
            self.self_attn = TAPAComplexMultiHeadAttention(
                d_model=d_model,
                dim_k=hid_dim,
                dim_v=hid_dim,
                n_heads=n_heads,
                theta=tapa_theta,
                alpha=tapa_alpha,
                modrelu_bias_init=modrelu_bias_init,
            )
        else:
            self.self_attn = PhaseAwareComplexMultiHeadAttention(
                d_model=d_model,
                dim_k=hid_dim,
                dim_v=hid_dim,
                n_heads=n_heads,
                modrelu_bias_init=modrelu_bias_init,
            )

        self.ffn = PhaseAwareComplexFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            modrelu_bias_init=modrelu_bias_init,
        )

        # Phase-aware layer norms (normalize magnitude, preserve phase)
        self.norm1 = PhaseAwareComplexLayerNorm(d_model, mode='magnitude')
        self.norm2 = PhaseAwareComplexLayerNorm(d_model, mode='magnitude')

        # Optional distance-based adjacency matrix
        self.use_adj_in_attn = use_adj_in_attn
        if use_adj_in_attn:
            self.loc_adj = DistanceAdj(gamma, bias)
        self.n_heads = n_heads

    def forward(self, real, imag):
        """
        Args:
            real, imag: [B, T, D] complex representation
        Returns:
            real, imag: [B, T, D] processed complex representation
        """
        # Compute adjacency if needed
        adj = None
        if self.use_adj_in_attn:
            adj = self.loc_adj(real.shape[0], real.shape[1], self.n_heads)

        # Self-attention + residual
        attn_real, attn_imag = self.self_attn(
            real, imag, real, imag, real, imag, adj=adj
        )
        real = real + attn_real
        imag = imag + attn_imag
        real, imag = self.norm1(real, imag)

        # FFN + residual
        ffn_real, ffn_imag = self.ffn(real, imag)
        real = real + ffn_real
        imag = imag + ffn_imag
        real, imag = self.norm2(real, imag)

        return real, imag


class PhaseAwareComplexAttention(nn.Module):
    """
    Complete Phase-Aware Complex Attention Module.

    This module replaces existing temporal attention modules with a phase-aware
    complex-valued approach using:
    1. Hilbert Projection: Learnable complex lift from real features
    2. ModReLU: Phase-preserving activation (replaces CReLU)
    3. Phase-aware attention: Complex attention scoring

    Processes audio and visual streams separately with their own Hilbert projections,
    then applies phase-aware complex attention to each.

    Key innovations:
    - Imaginary part learns temporal dynamics via orthogonal initialization
    - Phase is preserved through all operations (ModReLU, layer norm)
    - Temporal position can be encoded in the imaginary part

    For incremental learning: Phase coherence can be tracked to detect forgetting.
    """
    def __init__(
        self,
        sa_layer_num=1,
        xa_layer_num=1,  # Not used, kept for API compatibility
        feat_dim=1024,
        hid_dim=256,
        d_ff=512,
        head_num=4,
        dropout=0.1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
        # Phase-aware specific params
        use_temporal_bias=True,
        use_temporal_diff=True,  # Deprecated: use imag_mode instead
        imag_mode=None,  # 'temporal_diff', 'zero', 'learned' (recommended for TAPA)
        output_mode='real',  # 'real', 'magnitude', 'concat'
        modrelu_bias_init=-0.1,
        # TAPA parameters (Token-Aware Phase Attention)
        use_tapa=False,
        tapa_theta=0.1,
        tapa_alpha=0.5,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.output_mode = output_mode
        self.use_tapa = use_tapa

        # Determine imag_mode: explicit param takes priority, else use legacy use_temporal_diff
        if imag_mode is not None:
            self.imag_mode = imag_mode
        elif use_temporal_diff:
            self.imag_mode = 'temporal_diff'
        else:
            self.imag_mode = 'zero'

        # Hilbert Projections for audio and visual
        self.audio_hilbert = HilbertProjection(feat_dim, use_temporal_bias, imag_mode=self.imag_mode)
        self.visual_hilbert = HilbertProjection(feat_dim, use_temporal_bias, imag_mode=self.imag_mode)

        # Phase-aware transformer blocks for audio
        self.a_sa_blocks = nn.ModuleList([
            PhaseAwareComplexTransformerBlock(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=head_num,
                dropout=dropout,
                modrelu_bias_init=modrelu_bias_init,
                use_adj_in_attn=use_adj_in_attn,
                gamma=gamma,
                bias=bias,
                use_tapa=use_tapa,
                tapa_theta=tapa_theta,
                tapa_alpha=tapa_alpha,
            )
            for _ in range(sa_layer_num)
        ])

        # Phase-aware transformer blocks for visual
        self.v_sa_blocks = nn.ModuleList([
            PhaseAwareComplexTransformerBlock(
                d_model=feat_dim,
                hid_dim=hid_dim,
                d_ff=d_ff,
                n_heads=head_num,
                dropout=dropout,
                modrelu_bias_init=modrelu_bias_init,
                use_adj_in_attn=use_adj_in_attn,
                gamma=gamma,
                bias=bias,
                use_tapa=use_tapa,
                tapa_theta=tapa_theta,
                tapa_alpha=tapa_alpha,
            )
            for _ in range(sa_layer_num)
        ])

        # Output projections (if needed for concat mode)
        if output_mode == 'concat':
            self.audio_out_proj = nn.Linear(feat_dim * 2, feat_dim)
            self.visual_out_proj = nn.Linear(feat_dim * 2, feat_dim)

        # Learnable gates for residual (preserve original ImageBind features)
        self.audio_gate = nn.Parameter(torch.tensor(0.5))
        self.visual_gate = nn.Parameter(torch.tensor(0.5))

    def compute_phase_stats(self, real, imag):
        """Compute phase statistics for monitoring/analysis."""
        phase = torch.atan2(imag, real + 1e-8)  # [B, T, D]
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)

        # Phase coherence: how consistent phases are across features
        # Using circular statistics: coherence = |mean(e^(i*phase))|
        complex_phase = torch.exp(1j * phase.double())
        coherence = torch.abs(complex_phase.mean(dim=-1)).float()  # [B, T]

        return {
            'phase_raw': phase,  # [B, T, D] - RAW per-feature phases for proper MPC
            'phase': phase.mean(dim=-1),  # [B, T] avg phase (for logging only)
            'magnitude': magnitude.mean(dim=-1),  # [B, T] avg magnitude
            'coherence': coherence.mean(),  # scalar
        }

    def forward(self, a_tokens, v_tokens, return_phase_info=False):
        """
        Args:
            a_tokens: [B, T, D] audio features from ImageBind
            v_tokens: [B, T, D] visual features from ImageBind
            return_phase_info: if True, return phase statistics for analysis

        Returns:
            a_out: [B, T, D] processed audio features
            v_out: [B, T, D] processed visual features
            (optional) phase_info: dict with phase statistics
        """
        # Store originals for gated residual
        a_original = a_tokens
        v_original = v_tokens

        # =====================================================================
        # AUDIO STREAM
        # =====================================================================
        # Hilbert projection: real features -> complex (real + i*imag)
        audio_real, audio_imag = self.audio_hilbert(a_tokens)

        # NUMERICAL STABILITY: Replace NaN with zero (graceful recovery)
        if torch.isnan(audio_real).any() or torch.isnan(audio_imag).any():
            print("[WARNING] NaN detected after audio_hilbert, replacing with zeros")
            audio_real = torch.nan_to_num(audio_real, nan=0.0)
            audio_imag = torch.nan_to_num(audio_imag, nan=0.0)

        # Store pre-attention phase for analysis
        if return_phase_info:
            audio_phase_pre = self.compute_phase_stats(audio_real, audio_imag)

        # Apply phase-aware transformer blocks
        for blk_idx, blk in enumerate(self.a_sa_blocks):
            audio_real, audio_imag = blk(audio_real, audio_imag)
            # NUMERICAL STABILITY: Replace NaN with zero (graceful recovery)
            if torch.isnan(audio_real).any() or torch.isnan(audio_imag).any():
                print(f"[WARNING] NaN detected after audio block {blk_idx}, replacing with zeros")
                audio_real = torch.nan_to_num(audio_real, nan=0.0)
                audio_imag = torch.nan_to_num(audio_imag, nan=0.0)

        # Store post-attention phase
        if return_phase_info:
            audio_phase_post = self.compute_phase_stats(audio_real, audio_imag)

        # Output conversion based on mode
        if self.output_mode == 'real':
            audio_out = audio_real
        elif self.output_mode == 'magnitude':
            audio_out = torch.sqrt(audio_real**2 + audio_imag**2 + 1e-8)
        elif self.output_mode == 'concat':
            audio_out = self.audio_out_proj(torch.cat([audio_real, audio_imag], dim=-1))

        # Gated residual connection
        gate_a = torch.sigmoid(self.audio_gate)
        a_out = gate_a * a_original + (1 - gate_a) * audio_out

        # =====================================================================
        # VISUAL STREAM
        # =====================================================================
        visual_real, visual_imag = self.visual_hilbert(v_tokens)

        # NUMERICAL STABILITY: Replace NaN with zero (graceful recovery)
        if torch.isnan(visual_real).any() or torch.isnan(visual_imag).any():
            print("[WARNING] NaN detected after visual_hilbert, replacing with zeros")
            visual_real = torch.nan_to_num(visual_real, nan=0.0)
            visual_imag = torch.nan_to_num(visual_imag, nan=0.0)

        if return_phase_info:
            visual_phase_pre = self.compute_phase_stats(visual_real, visual_imag)

        for blk_idx, blk in enumerate(self.v_sa_blocks):
            visual_real, visual_imag = blk(visual_real, visual_imag)
            # NUMERICAL STABILITY: Replace NaN with zero (graceful recovery)
            if torch.isnan(visual_real).any() or torch.isnan(visual_imag).any():
                print(f"[WARNING] NaN detected after visual block {blk_idx}, replacing with zeros")
                visual_real = torch.nan_to_num(visual_real, nan=0.0)
                visual_imag = torch.nan_to_num(visual_imag, nan=0.0)

        if return_phase_info:
            visual_phase_post = self.compute_phase_stats(visual_real, visual_imag)

        # Output conversion
        if self.output_mode == 'real':
            visual_out = visual_real
        elif self.output_mode == 'magnitude':
            visual_out = torch.sqrt(visual_real**2 + visual_imag**2 + 1e-8)
        elif self.output_mode == 'concat':
            visual_out = self.visual_out_proj(torch.cat([visual_real, visual_imag], dim=-1))

        # Gated residual
        gate_v = torch.sigmoid(self.visual_gate)
        v_out = gate_v * v_original + (1 - gate_v) * visual_out

        # =====================================================================
        # Return
        # =====================================================================
        if torch.isnan(a_out).any() or torch.isnan(v_out).any():
            pdb.set_trace()

        if return_phase_info:
            phase_info = {
                # Raw per-feature phases for proper MPC computation [B, T, D]
                'audio_phase_raw': audio_phase_post['phase_raw'],
                'visual_phase_raw': visual_phase_post['phase_raw'],
                # Averaged phases for logging [B, T]
                'audio_phase_pre': audio_phase_pre['phase'],
                'audio_phase_post': audio_phase_post['phase'],
                'audio_coherence_pre': audio_phase_pre['coherence'],
                'audio_coherence_post': audio_phase_post['coherence'],
                'visual_phase_pre': visual_phase_pre['phase'],
                'visual_phase_post': visual_phase_post['phase'],
                'visual_coherence_pre': visual_phase_pre['coherence'],
                'visual_coherence_post': visual_phase_post['coherence'],
                'audio_gate': gate_a.item(),
                'visual_gate': gate_v.item(),
            }
            return a_out, v_out, phase_info

        return a_out, v_out


# ============================================================================
# Mixture-of-Experts Temporal Attention Module (Boundary-Aware)
# ============================================================================

class TemporalExpert(nn.Module):
    """
    A single temporal expert with a specific inductive bias.

    expert_type controls the temporal operator:
      - 'steady':   wide causal+future context via Conv1d (k=5) — for interior frames
      - 'onset':    causal temporal difference f[t] - f[t-1] + Conv1d (k=3) — detects event start
      - 'offset':   anti-causal difference f[t] - f[t+1] + Conv1d (k=3) — detects event end
      - 'boundary': narrow Conv1d (k=1) on raw features — preserves sharp per-frame detail
    """
    def __init__(self, feat_dim=1024, hidden_dim=256, expert_type='steady', dropout=0.1):
        super().__init__()
        self.expert_type = expert_type
        self.feat_dim = feat_dim

        if expert_type == 'steady':
            # Wide temporal context: captures sustained event interiors
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(feat_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, feat_dim, kernel_size=1),
            )
        elif expert_type == 'onset':
            # Causal diff + narrow conv: sensitive to rising edges
            self.diff_proj = nn.Linear(feat_dim, feat_dim)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(feat_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, feat_dim, kernel_size=1),
            )
        elif expert_type == 'offset':
            # Anti-causal diff + narrow conv: sensitive to falling edges
            self.diff_proj = nn.Linear(feat_dim, feat_dim)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(feat_dim * 2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, feat_dim, kernel_size=1),
            )
        elif expert_type == 'boundary':
            # Pointwise: no temporal mixing, preserves frame-level sharpness
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(feat_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, feat_dim, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] temporal features
        Returns:
            out: [B, T, D] expert-processed features (residual added outside)
        """
        B, T, D = x.shape

        if self.expert_type == 'onset':
            # Causal diff: f[t] - f[t-1], zero-padded at t=0
            diff = x[:, 1:, :] - x[:, :-1, :]                     # [B, T-1, D]
            diff = torch.cat([torch.zeros(B, 1, D, device=x.device), diff], dim=1)  # [B, T, D]
            diff = self.diff_proj(diff)
            combined = torch.cat([x, diff], dim=-1)                 # [B, T, 2D]
            out = self.temporal_conv(combined.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        elif self.expert_type == 'offset':
            # Anti-causal diff: f[t] - f[t+1], zero-padded at t=T-1
            diff = x[:, :-1, :] - x[:, 1:, :]                     # [B, T-1, D]
            diff = torch.cat([diff, torch.zeros(B, 1, D, device=x.device)], dim=1)  # [B, T, D]
            diff = self.diff_proj(diff)
            combined = torch.cat([x, diff], dim=-1)                 # [B, T, 2D]
            out = self.temporal_conv(combined.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        else:
            # 'steady' and 'boundary': just conv on features
            out = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        return out


class MoERouter(nn.Module):
    """
    Learned router that produces per-frame expert weights.

    Input: concatenated audio + visual features (optionally with temporal diffs).
    Output: [B, T, num_experts] soft routing weights.
    """
    def __init__(self, input_dim, hidden_dim=128, num_experts=4, use_temporal_diff=True):
        super().__init__()
        self.use_temporal_diff = use_temporal_diff
        self.num_experts = num_experts

        # If using temporal diff, router input = 2 * input_dim (features + temporal gradient magnitude)
        router_input_dim = input_dim * 2 if use_temporal_diff else input_dim

        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, top_k=0):
        """
        Args:
            features: [B, T, D] concatenated audio+visual features
            top_k: if > 0, only keep top-k experts (sparse routing). 0 = soft (all experts).
        Returns:
            routing_weights: [B, T, num_experts] softmax weights
        """
        B, T, D = features.shape

        if self.use_temporal_diff:
            # Temporal gradient magnitude as routing signal
            diff = features[:, 1:, :] - features[:, :-1, :]  # [B, T-1, D]
            diff = torch.cat([torch.zeros(B, 1, D, device=features.device), diff], dim=1)
            diff_magnitude = diff.abs()  # magnitude of change per frame
            router_input = torch.cat([features, diff_magnitude], dim=-1)  # [B, T, 2D]
        else:
            router_input = features

        logits = self.router(router_input)  # [B, T, num_experts]

        if top_k > 0 and top_k < self.num_experts:
            # Sparse routing: zero out non-top-k experts
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))

        routing_weights = F.softmax(logits, dim=-1)  # [B, T, num_experts]
        return routing_weights


class MoETemporalAttentionModule(nn.Module):
    """
    Mixture-of-Experts Temporal Attention for boundary-aware audio-visual processing.

    Architecture:
        1. Audio and visual features are processed by separate sets of MoE experts
        2. Each expert has a different temporal inductive bias:
           - Steady-state: wide context for interior frames
           - Onset: causal diff for detecting event starts
           - Offset: anti-causal diff for detecting event ends
           - Boundary-refine: pointwise for sharp frame-level decisions
        3. A shared router sees concatenated AV features + temporal gradients
           and produces per-frame, per-expert weights
        4. Final output = weighted sum of expert outputs + residual

    Interface matches TemporalAttentionModule:
        forward(a_tokens, v_tokens) -> (a_tokens, v_tokens)

    References:
        - AVMoE (NeurIPS 2024): MoE adapters for audio-visual tasks
        - DyBDet (ECCV 2024): dynamic computation for boundary detection
        - TriDet (CVPR 2023): start/end/center specialized branches for TAL
    """

    EXPERT_TYPES = ['steady', 'onset', 'offset', 'boundary']

    def __init__(
        self,
        feat_dim=1024,
        expert_hidden_dim=256,
        router_hidden_dim=128,
        num_experts=4,
        dropout=0.1,
        top_k=0,
        use_temporal_diff_routing=True,
        load_balance_weight=0.01,
        # Self-attention params (shared across experts, applied before routing)
        sa_hid_dim=256,
        sa_ff_dim=512,
        sa_head_num=1,
        sa_layer_num=1,
        use_adj_in_attn=False,
        gamma=0.6,
        bias=0.2,
        use_mask_in_attn=False,
        win_size=1,
        norm_flag=None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight

        # Shared self-attention: global temporal context BEFORE expert routing
        # Each modality gets its own self-attention blocks (unshared weights)
        sa_block_a = TransformerEncoder(
            d_model=feat_dim, hid_dim=sa_hid_dim, d_ff=sa_ff_dim,
            n_heads=sa_head_num, dropout=dropout,
            use_adj_in_attn=use_adj_in_attn, gamma=gamma, bias=bias,
            use_mask_in_attn=use_mask_in_attn, win_size=win_size, norm_flag=norm_flag,
        )
        sa_block_v = TransformerEncoder(
            d_model=feat_dim, hid_dim=sa_hid_dim, d_ff=sa_ff_dim,
            n_heads=sa_head_num, dropout=dropout,
            use_adj_in_attn=use_adj_in_attn, gamma=gamma, bias=bias,
            use_mask_in_attn=use_mask_in_attn, win_size=win_size, norm_flag=norm_flag,
        )
        self.audio_sa_blocks = nn.Sequential(*[dcp(sa_block_a) for _ in range(sa_layer_num)])
        self.visual_sa_blocks = nn.Sequential(*[dcp(sa_block_v) for _ in range(sa_layer_num)])

        # Determine expert types: cycle through EXPERT_TYPES if num_experts != 4
        expert_types = []
        for i in range(num_experts):
            expert_types.append(self.EXPERT_TYPES[i % len(self.EXPERT_TYPES)])

        # Audio experts
        self.audio_experts = nn.ModuleList([
            TemporalExpert(feat_dim, expert_hidden_dim, etype, dropout)
            for etype in expert_types
        ])

        # Visual experts
        self.visual_experts = nn.ModuleList([
            TemporalExpert(feat_dim, expert_hidden_dim, etype, dropout)
            for etype in expert_types
        ])

        # Shared router: sees concatenated audio + visual features (after self-attention)
        self.router = MoERouter(
            input_dim=feat_dim * 2,  # concat audio + visual
            hidden_dim=router_hidden_dim,
            num_experts=num_experts,
            use_temporal_diff=use_temporal_diff_routing,
        )

        self.top_k = top_k

        # Per-modality output layer norms
        self.audio_norm = nn.LayerNorm(feat_dim)
        self.visual_norm = nn.LayerNorm(feat_dim)

        # Learnable residual gates (initialized to let residual dominate early)
        self.audio_gate = nn.Parameter(torch.tensor(0.0))
        self.visual_gate = nn.Parameter(torch.tensor(0.0))

        self.expert_types = expert_types

    def compute_load_balance_loss(self, routing_weights):
        """
        Encourage balanced expert utilization via KL divergence.

        Args:
            routing_weights: [B, T, num_experts]
        Returns:
            loss: scalar
        """
        # Average routing probability across batch and time
        expert_probs_mean = routing_weights.mean(dim=(0, 1))  # [num_experts]
        # Target: uniform distribution
        uniform = torch.ones_like(expert_probs_mean) / self.num_experts
        # KL(uniform || expert_probs) — encourages balanced usage
        loss = F.kl_div(
            expert_probs_mean.log().unsqueeze(0),
            uniform.unsqueeze(0),
            reduction='batchmean',
            log_target=False,
        )
        return loss

    def forward(self, a_tokens, v_tokens, return_routing_info=False):
        """
        Args:
            a_tokens: [B, T, 1024] audio features
            v_tokens: [B, T, 1024] visual features
            return_routing_info: if True, return dict with routing weights + load balance loss

        Returns:
            a_tokens: [B, T, 1024] processed audio
            v_tokens: [B, T, 1024] processed visual
            (optional) routing_info: dict with 'routing_weights', 'load_balance_loss', 'expert_types'
        """
        # Step 1: Shared self-attention for global temporal context
        for blk in self.audio_sa_blocks:
            a_tokens = blk(a_tokens, a_tokens, a_tokens)
        for blk in self.visual_sa_blocks:
            v_tokens = blk(v_tokens, v_tokens, v_tokens)

        a_residual = a_tokens
        v_residual = v_tokens

        # Step 2: Router sees self-attended features
        router_input = torch.cat([a_tokens, v_tokens], dim=-1)  # [B, T, 2D]
        routing_weights = self.router(router_input, top_k=self.top_k)  # [B, T, num_experts]

        # Process each expert and accumulate weighted outputs
        a_out = torch.zeros_like(a_tokens)
        v_out = torch.zeros_like(v_tokens)

        for i, (a_expert, v_expert) in enumerate(zip(self.audio_experts, self.visual_experts)):
            w_i = routing_weights[:, :, i].unsqueeze(-1)  # [B, T, 1]
            a_out = a_out + w_i * a_expert(a_tokens)
            v_out = v_out + w_i * v_expert(v_tokens)

        # Gated residual connection
        gate_a = torch.sigmoid(self.audio_gate)
        gate_v = torch.sigmoid(self.visual_gate)
        a_tokens = gate_a * a_out + (1 - gate_a) * a_residual
        v_tokens = gate_v * v_out + (1 - gate_v) * v_residual

        # Layer norm
        a_tokens = self.audio_norm(a_tokens)
        v_tokens = self.visual_norm(v_tokens)

        if torch.isnan(a_tokens).any() or torch.isnan(v_tokens).any():
            pdb.set_trace()

        if return_routing_info:
            lb_loss = self.compute_load_balance_loss(routing_weights)
            routing_info = {
                'routing_weights': routing_weights.detach(),  # [B, T, num_experts]
                'load_balance_loss': lb_loss * self.load_balance_weight,
                'expert_types': self.expert_types,
            }
            return a_tokens, v_tokens, routing_info

        return a_tokens, v_tokens