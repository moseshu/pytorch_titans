from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

# einstein notation related

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

# hyper connections / attend from x-transformers, which handles different queries and key lengths better

from x_transformers.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

# proposed neural memory

from neural_memory import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        sliding = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.attend = Attend(causal = True, **attend_kwargs)

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding # sliding window attn - doubt their non-sliding results being the best. local attention with overlapping windows is very strong

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        batch = token.shape[0]

        # attention

        token = self.norm(token)

        q, k, v = self.to_qkv(token).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        # caching

        ck, cv = cache
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)

        next_cache = (k, v)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h n d -> b h n d') for t in (q, k, v))

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = (k, v)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, self.num_persist_mem_tokens, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # attention

        out = flex_attn_fn(q, k, v)

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            assert seq.shape[-2] == 1
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache)

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        # auto pad to multiple

        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch = False)

        # attention

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # value residual

        orig_v = v

        if exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # caching

        next_cache = tuple(map(inverse_segment, (k, v)))

        # relative positions

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold

        q, k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        # maybe sliding for cpu

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value = 0., dim = 1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim = -2)
            v = cat((v[:, :-1], v[:, 1:]), dim = -2)
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            # take care of masking

            idx = torch.arange(seq.shape[-2], device = seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value = True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # persistent memory

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)

        # attention

        out, _ = self.attend(q, k, v, **attend_kwargs)

        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        out = inverse_segment(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)


    
import torch
from torch import nn
import torch.nn.functional as F

class LightweightLatentAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LatentReasoningBlock(nn.Module):
    """
    A light-weight latent world model that performs three operations:
    1) inner-loop refinement of a thought state (System-2 style updates)
    2) latent roll-outs that predict multiple future states
    3) gating to decide how much of the deliberate state should influence the stream
    """
    def __init__(
        self,
        dim,
        depth: int = 2,
        expansion_factor: float = 2.,
        inner_steps: int = 2,
        future_horizon: int = 3,
        use_inner_attention: bool = True,
        inner_attn_heads: int = 4,
        inner_attn_dim_head: int = 32,
        use_act: bool = True,
        act_epsilon: float = 1e-2,
        act_loss_weight: float = 0.01
    ):
        super().__init__()
        self.depth = depth
        self.inner_steps = inner_steps
        self.future_horizon = future_horizon
        self.use_inner_attention = use_inner_attention
        self.use_act = use_act and inner_steps > 0
        self.act_epsilon = act_epsilon
        self.act_loss_weight = act_loss_weight

        hidden_dim = int(dim * expansion_factor)

        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim * 2),
                GEGLU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(depth)
        ])

        self.inner_norm = nn.LayerNorm(dim) if inner_steps > 0 else None

        self.inner_cell = None
        self.inner_attn = None
        self.inner_attn_norm = None

        if inner_steps > 0:
            if use_inner_attention:
                self.inner_attn_norm = nn.LayerNorm(dim)
                self.inner_attn = LightweightLatentAttention(
                    dim,
                    heads = inner_attn_heads,
                    dim_head = inner_attn_dim_head
                )
            else:
                self.inner_cell = nn.GRUCell(dim, dim)

        self.halt_proj = None
        if self.use_act:
            self.halt_proj = nn.Linear(dim, 1)

        self.future_predictor = None
        if future_horizon > 0:
            self.future_predictor = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim * future_horizon)
            )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        current_input,
        retrieved_memory,
        future_targets = None,
        future_target_mask = None
    ):
        """
        current_input: 当前 residual stream 的内容 [B, N, D]
        retrieved_memory: Neural Memory 返回的上下文 [B, N, D]
        future_targets: 监督未来 latent 的 teacher 信号 [B, H, N, D]
        future_target_mask: 哪些位置存在真实未来 [B, H, N]
        """

        # 合并输入得到初始思考状态
        thought_state = current_input + retrieved_memory
        scratchpad_states = []

        # 多层非线性推理
        for layer in self.reasoning_layers:
            thought_state = thought_state + layer(thought_state)
            scratchpad_states.append(thought_state)

        # 内循环：在同一个时间步上反复思考
        act_loss = None
        if self.inner_steps > 0 and (exists(self.inner_cell) or exists(self.inner_attn)):
            inner_state = thought_state
            iterative_states = []
            flat_dim = current_input.shape[-1]
            gru_context = None

            if exists(self.inner_cell):
                gru_context = (current_input + retrieved_memory).reshape(-1, flat_dim)

            if self.use_act:
                batch, seq_len = inner_state.shape[:2]
                device = inner_state.device
                halting_prob = torch.zeros(batch, seq_len, device = device, dtype = inner_state.dtype)
                updates_accum = torch.zeros_like(halting_prob)
                running_mask = torch.ones_like(halting_prob, dtype = torch.bool)
                accumulated_state = torch.zeros_like(inner_state)

            for _ in range(self.inner_steps):
                iterative_states.append(inner_state)

                if exists(self.inner_attn):
                    attn_in = self.inner_attn_norm(inner_state)
                    next_state = inner_state + self.inner_attn(attn_in)
                elif exists(self.inner_cell):
                    state_flat = inner_state.reshape(-1, flat_dim)
                    state_flat = self.inner_cell(gru_context, state_flat)
                    next_state = state_flat.view_as(inner_state)
                else:
                    break

                next_state = self.inner_norm(next_state)

                if self.use_act:
                    halt = torch.sigmoid(self.halt_proj(inner_state)).squeeze(-1)
                    halt = halt * running_mask.float()
                    new_halting = halting_prob + halt
                    overflow = torch.clamp(new_halting - 1., min = 0.)
                    halt_adjusted = halt - overflow
                    halting_prob = halting_prob + halt_adjusted

                    accumulated_state = accumulated_state + inner_state * halt_adjusted.unsqueeze(-1)
                    updates_accum = updates_accum + running_mask.float()

                    running_mask = halting_prob < (1. - self.act_epsilon)
                    inner_state = torch.where(running_mask.unsqueeze(-1), next_state, inner_state)

                    if not running_mask.any():
                        break
                else:
                    inner_state = next_state

            if self.use_act:
                remainder = (1. - halting_prob).clamp(min = 0.)
                accumulated_state = accumulated_state + inner_state * remainder.unsqueeze(-1)
                ponder_cost = updates_accum + remainder
                act_loss = ponder_cost.float().mean() * self.act_loss_weight
                inner_state = accumulated_state

            thought_state = inner_state
            scratchpad_states.extend(iterative_states)

        # 预测未来 latent
        rollout_loss = None
        future_preds = None

        if exists(self.future_predictor) and (self.training or exists(future_targets)):
            future_preds = self.future_predictor(thought_state)
            future_preds = rearrange(
                future_preds,
                'b n (h d) -> b h n d',
                h = self.future_horizon
            )

            if exists(future_targets):
                rollout_loss = F.smooth_l1_loss(
                    future_preds.float(),
                    future_targets.float(),
                    reduction = 'none'
                )

                if exists(future_target_mask):
                    mask = future_target_mask.unsqueeze(-1).type_as(rollout_loss)
                    mask = mask.expand_as(rollout_loss)
                    rollout_loss = rollout_loss * mask
                    denom = mask.sum().clamp(min = 1.)
                    rollout_loss = rollout_loss.sum() / denom
                else:
                    rollout_loss = rollout_loss.mean()

        scratchpad = None
        if len(scratchpad_states) > 0:
            scratchpad = torch.stack(scratchpad_states, dim = 1)

        combined = torch.cat([current_input, thought_state], dim = -1)
        g = self.gate(combined)
        enhanced_state = g * thought_state + (1 - g) * current_input

        return enhanced_state, {
            'future_predictions': future_preds,
            'rollout_loss': rollout_loss,
            'scratchpad': scratchpad,
            'act_loss': act_loss.float() if exists(act_loss) else None,
            'gate_mean': g.detach().float().mean(),
            'thought_norm': thought_state.detach().float().norm(dim = -1).mean(),
            'future_pred_norm': future_preds.detach().float().norm(dim = -1).mean() if exists(future_preds) else None,
            'future_target_norm': future_targets.detach().float().norm(dim = -1).mean() if exists(future_targets) else None
        }
    
    
# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        neural_memory_segment_len = None,
        neural_mem_gate_attn_output = False,
        neural_memory_add_value_residual = False,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        neural_memory_batch_size = None,
        neural_memory_qkv_receives_diff_views = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        use_flex_attn = False,
        sliding_window_attn = False,
        neural_mem_weight_residual = False,
        latent_reasoning: bool = True,
        latent_reasoning_kwargs: dict | None = None,
        reasoning_loss_weight: float = 0.1,
        latent_reasoning_eval_targets: bool = False,
        token_emb: Module | None = None,
    ):
        super().__init__()

        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb
        latent_reasoning_kwargs = default(latent_reasoning_kwargs, dict())

        self.reasoning_module = None
        self.reasoning_loss_weight = reasoning_loss_weight
        self.reasoning_horizon = 0
        self.reasoning_eval_targets = latent_reasoning_eval_targets

        if latent_reasoning:
            self.reasoning_module = LatentReasoningBlock(dim, **latent_reasoning_kwargs)
            self.reasoning_horizon = latent_reasoning_kwargs.get('future_horizon', 0)

        # absolute positions
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim = dim, num_axial_dims = 2)

        # long term mem tokens

        self.segment_len = segment_len

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0

        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)

        # maybe sliding window attn

        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        # hyper connection

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        layers = tuple(range(1, depth + 1))

        neural_memory_layers = default(neural_memory_layers, layers)

        # weight residual related

        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True

        # mem, attn, and feedforward layers

        for layer in layers:
            is_first = layer == 1

            # attention and feedforward

            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                use_flex_attn = use_flex_attn,
                accept_value_residual = not is_first,
                num_longterm_mem_tokens = num_longterm_mem_tokens,
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )

            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)

                if not is_first and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1 # for each layer, have memory input select from attn inp, attn out, ff inp, and ff out - plus one for the current point in the residual stream (memory input)

                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )

                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )

                is_first_neural_mem = False

            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(),
                init_hyper_conn(),
                mem_qkv_layer_selector,
                mem,
                attn,
                ff,
            ]))

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # whether to gate the attention output with the retrieved memories

        self.gate_attn_output = neural_mem_gate_attn_output

        # zero for maybe aux loss + device

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.num_persist_mem_tokens = num_persist_mem_tokens

    def seq_index_is_longterm(
        self,
        seq_index
    ):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    def _build_future_targets(self, states):
        horizon = self.reasoning_horizon

        if horizon <= 0:
            return None, None

        states = states.detach()
        batch, seq_len, dim = states.shape
        device, dtype = states.device, states.dtype

        if self.num_longterm_mem_tokens > 0:
            total_segment_len, segment_len = self.attn_window_size, self.segment_len
            seq_idx = torch.arange(seq_len, device = device)
            is_longterm = (seq_idx % total_segment_len) >= segment_len
            base_mask = (~is_longterm).unsqueeze(0).expand(batch, -1)
        else:
            base_mask = torch.ones(batch, seq_len, device = device, dtype = torch.bool)

        future_targets = []
        future_masks = []

        for step in range(1, horizon + 1):
            future = torch.zeros(batch, seq_len, dim, device = device, dtype = dtype)
            mask = base_mask.clone()

            if step < seq_len:
                future[:, :-step] = states[:, step:]
                mask[:, :-step] = mask[:, :-step] & base_mask[:, step:]
            else:
                mask[:] = False

            mask[:, -step:] = False

            future_targets.append(future)
            future_masks.append(mask)

        future_targets = torch.stack(future_targets, dim = 1)
        future_masks = torch.stack(future_masks, dim = 1)

        return future_targets, future_masks

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(
            min_p = 0.1,
        ),
        show_progress = True,
        use_cache = False
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        # cache for axial pos, attention, and neural memory

        cache = None
        factorized_pos_emb = None

        # precompute factorized pos emb

        if use_cache:
            seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

            axial_dims = self.axial_pos_emb.maybe_derive_outer_dim(seq_len_with_mem, (self.neural_memory_segment_len,))

            factorized_pos_emb = self.axial_pos_emb(axial_dims, return_factorized = True)

        # sample

        with tqdm.tqdm(total = sample_num_times, disable = not show_progress) as pbar:

            while out.shape[-1] < seq_len:

                logits, next_cache = self.forward(
                    out,
                    disable_flex_attn = True,
                    cache = cache,
                    return_cache = True,
                    factorized_pos_emb = factorized_pos_emb
                )

                if use_cache:
                    cache = next_cache

                if not exists(logits):
                    continue

                logits = logits[:, -1]

                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature = temperature)

                out = torch.cat((out, sample), dim = -1)
                pbar.update(1)

        self.train(was_training)

        return out[..., prompt_seq_len:]

    def forward(
        self,
        x,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False,
        cache = None,
        return_cache = False,
        factorized_pos_emb = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # math

        batch, seq_len, neural_mem_segment_len, segment_len, num_longterm_mem_tokens, attn_window_size = *x.shape, self.neural_memory_segment_len, self.segment_len, self.num_longterm_mem_tokens, self.attn_window_size

        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # token embedding

        x = self.token_emb(x)

        # intersperse longterm memory

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad = False)

        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')

        x = inverse_segment(x)

        # splice out unneeded tokens from padding for longterm mems

        x = x[:, :seq_len_with_mem]

        # apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network

        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)

        x = x + pos_emb

        # prep flex attention

        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # kv caching

        is_inferencing = exists(cache)

        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)

        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # value residual

        value_residual = None

        # neural mem weight residual

        mem_weight_residual = None

        # layers for the neural mem to select the qkv inputs from

        mem_input_layers = []
        reasoning_aux_loss = None
        reasoning_rollout_loss = None
        reasoning_act_loss = None
        reasoning_gate_mean = None
        reasoning_thought_norm = None
        reasoning_future_pred_norm = None
        reasoning_future_target_norm = None
        reasoning_stat_count = 0

        # when inferencing, only do one token at a time

        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        # expand and reduce streams for hyper connections

        x = self.expand_streams(x)

        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # maybe neural memory

            if exists(mem):

                mem_input, add_residual = mem_hyper_conn(x)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))

                    # let the current `mem_input` select the 3 layers for qkv

                    selected = mem_qkv_layer_selector(mem_input)

                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                retrieved, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state = next(neural_mem_caches, None),
                    prev_weights = mem_weight_residual
                )

                context_to_add = retrieved

                if exists(self.reasoning_module):
                    future_targets = future_target_mask = None

                    if (self.training or self.reasoning_eval_targets) and self.reasoning_horizon > 0:
                        future_targets, future_target_mask = self._build_future_targets(mem_input)

                    context_to_add, reasoning_info = self.reasoning_module(
                        mem_input,
                        retrieved,
                        future_targets = future_targets,
                        future_target_mask = future_target_mask
                    )

                    rollout_loss = reasoning_info.get('rollout_loss')
                    act_loss = reasoning_info.get('act_loss')

                    if exists(rollout_loss):
                        reasoning_rollout_loss = rollout_loss if not exists(reasoning_rollout_loss) else (reasoning_rollout_loss + rollout_loss)
                    if exists(act_loss):
                        reasoning_act_loss = act_loss if not exists(reasoning_act_loss) else (reasoning_act_loss + act_loss)

                    for aux_loss in (rollout_loss, act_loss):
                        if exists(aux_loss):
                            reasoning_aux_loss = aux_loss if not exists(reasoning_aux_loss) else (reasoning_aux_loss + aux_loss)

                    gate_mean = reasoning_info.get('gate_mean')
                    thought_norm = reasoning_info.get('thought_norm')
                    future_pred_norm = reasoning_info.get('future_pred_norm')
                    future_target_norm = reasoning_info.get('future_target_norm')

                    if exists(gate_mean):
                        reasoning_gate_mean = gate_mean if not exists(reasoning_gate_mean) else (reasoning_gate_mean + gate_mean)
                        reasoning_stat_count += 1
                    if exists(thought_norm):
                        reasoning_thought_norm = thought_norm if not exists(reasoning_thought_norm) else (reasoning_thought_norm + thought_norm)
                    if exists(future_pred_norm):
                        reasoning_future_pred_norm = future_pred_norm if not exists(reasoning_future_pred_norm) else (reasoning_future_pred_norm + future_pred_norm)
                    if exists(future_target_norm):
                        reasoning_future_target_norm = future_target_norm if not exists(reasoning_future_target_norm) else (reasoning_future_target_norm + future_target_norm)

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = context_to_add.sigmoid()
                else:
                    x = add_residual(context_to_add)

            # attention

            attn_in, add_residual = attn_hyper_conn(x)

            mem_input_layers.append(attn_in)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_out_gates,
                cache = next(kv_caches, None)
            )

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)

            x = add_residual(attn_out)

            # caches

            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # feedforward

            ff_in, add_ff_residual = ff_hyper_conn(x)

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)

            mem_input_layers.append(ff_out)

            x = add_ff_residual(ff_out)

        # taking care of cache first
        # for early return when processing long term mem tokens during inference

        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])

            # handle kv cache length depending on local attention type

            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]

            kv_cache_length = next_kv_caches.shape[-2]

            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1,
                next_kv_caches,
                next_neural_mem_caches
            )

            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)

            if is_inferencing and is_longterm_mem:
                return None, next_cache

        # hyper connection reducing of streams

        x = self.reduce_streams(x)

        # excise out the memories

        if not is_inferencing:

            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad = False)

            x, _ = inverse_pack_mems(x)

            x = inverse_segment(x)

            x = x[:, :seq_len]

        # to logits

        x = self.norm(x)

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        lm_loss = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
        total_loss = lm_loss

        if exists(reasoning_aux_loss) and self.reasoning_loss_weight > 0.:
            total_loss = total_loss + self.reasoning_loss_weight * reasoning_aux_loss

        if return_loss_breakdown:
            zero_like_loss = self.zero.to(device = lm_loss.device, dtype = lm_loss.dtype)
            wm_loss = reasoning_aux_loss * self.reasoning_loss_weight if exists(reasoning_aux_loss) else zero_like_loss
            wm_raw = reasoning_aux_loss.detach() if exists(reasoning_aux_loss) else zero_like_loss
            wm_rollout_raw = reasoning_rollout_loss.detach() if exists(reasoning_rollout_loss) else zero_like_loss
            wm_act_raw = reasoning_act_loss.detach() if exists(reasoning_act_loss) else zero_like_loss

            if reasoning_stat_count > 0 and exists(reasoning_gate_mean):
                reasoning_gate_mean_out = (reasoning_gate_mean / float(reasoning_stat_count)).detach()
            else:
                reasoning_gate_mean_out = zero_like_loss

            reasoning_thought_norm_out = (reasoning_thought_norm / float(reasoning_stat_count)).detach() if reasoning_stat_count > 0 and exists(reasoning_thought_norm) else zero_like_loss
            reasoning_future_pred_norm_out = (reasoning_future_pred_norm / float(reasoning_stat_count)).detach() if reasoning_stat_count > 0 and exists(reasoning_future_pred_norm) else zero_like_loss
            reasoning_future_target_norm_out = (reasoning_future_target_norm / float(reasoning_stat_count)).detach() if reasoning_stat_count > 0 and exists(reasoning_future_target_norm) else zero_like_loss

            breakdown = dict(
                lm = lm_loss.detach(),
                world_model = wm_loss.detach(),
                world_model_raw = wm_raw,
                world_model_rollout_raw = wm_rollout_raw,
                world_model_act_raw = wm_act_raw,
                reasoning_gate_mean = reasoning_gate_mean_out,
                reasoning_thought_norm = reasoning_thought_norm_out,
                reasoning_future_pred_norm = reasoning_future_pred_norm_out,
                reasoning_future_target_norm = reasoning_future_target_norm_out
            )
            return total_loss, breakdown

        return total_loss
