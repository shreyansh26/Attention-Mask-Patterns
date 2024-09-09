import torch
from triton.testing import do_bench
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, create_mask, and_masks

import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from utils import visualize_attention_scores, plot_timing_graph

torch.set_default_device('cuda')

B = 8
H = 16
S = 4096
D = 64
SLIDING_WINDOW = 256

def bidirectional_sliding_window(b, h, q_idx, kv_idx):
    windowed_mask = (q_idx - kv_idx).abs() <= SLIDING_WINDOW
    return windowed_mask

q, k, v = [torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float16) for _ in range(3)]

mask_mod = bidirectional_sliding_window
block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
mask = create_mask(mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S)

print("Bidirectional sliding window mask:", block_mask[0])

# Benchmark and Correctness
flex_attention = torch.compile(flex_attention)

print("Flex Attention:", flex_attention(q, k, v, block_mask=block_mask).sum())
flex_attention_fwd_time = do_bench(lambda: flex_attention(q, k, v, block_mask=block_mask).sum())
flex_attention_bwd_time = do_bench(lambda: flex_attention(q, k, v, block_mask=block_mask).sum().backward())
print("Flex Attention fwd:", flex_attention_fwd_time)
print("Flex Attention bwd:", flex_attention_bwd_time)

print("xformers/sdpa with mask:", scaled_dot_product_attention(q, k, v, attn_mask=mask).sum())
xformers_sdpa_with_mask_fwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v, attn_mask=mask).sum())
xformers_sdpa_with_mask_bwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v, attn_mask=mask).sum().backward())
print("xformers/sdpa with mask fwd:", xformers_sdpa_with_mask_fwd_time)
print("xformers/sdpa with mask bwd:", xformers_sdpa_with_mask_bwd_time)

# Only enable flash attention backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    print("FA (full):", scaled_dot_product_attention(q, k, v).sum())
    fa_fwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v).sum())
    fa_bwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v).sum().backward())
    print("FA (full) fwd:", fa_fwd_time)
    print("FA (full) bwd:", fa_bwd_time)

# Plot timing
name = "bidirectional_sliding_window"
plot_timing_graph([flex_attention_fwd_time, flex_attention_bwd_time, "FlexAttention"], 
                [xformers_sdpa_with_mask_fwd_time, xformers_sdpa_with_mask_bwd_time, "xFormers/SDPA + mask"], 
                [fa_fwd_time, fa_bwd_time, "FA (Full)"], seq_length=S,
                name=name, path=Path(f"plots/{name}/timing.png"))

# Plot mask
B = 1
H = 1
S = 32
D = 8
SLIDING_WINDOW = 4

q, k = [torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(2)]
mask_mod = bidirectional_sliding_window

visualize_attention_scores(q, k, mask_mod=mask_mod, name=name, path=Path(f"plots/{name}/mask.png"))