import torch
from triton.testing import do_bench
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, create_mask, or_masks

import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from utils import visualize_attention_scores, plot_timing_graph

torch.set_default_device('cuda')

B = 8
H = 16
S = 32768
D = 64
DOCUMENT_IDXS = torch.zeros(S, dtype=torch.int, device='cuda')
DOCUMENT_IDXS[:4096] = 0
DOCUMENT_IDXS[4096:8192] = 1
for i in range(8192, S, 8192):
    DOCUMENT_IDXS[i : i + 8192] = i // 8192 + 1

def multi_document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = (q_idx >= kv_idx)
    document_mask = (DOCUMENT_IDXS[q_idx] == DOCUMENT_IDXS[kv_idx])
    return causal_mask & document_mask

q, k, v = [torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float16) for _ in range(3)]

mask_mod = multi_document_causal_mask
block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
mask = create_mask(mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S)

print("Multi-document causal mask:", block_mask[0])

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
    print("FA (causal):", scaled_dot_product_attention(q, k, v, is_causal=True).sum())
    fa_fwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v, is_causal=True).sum())
    fa_bwd_time = do_bench(lambda: scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward())
    print("FA (causal) fwd:", fa_fwd_time)
    print("FA (causal) bwd:", fa_bwd_time)

# Plot timing
name = "multi_document_causal_mask"
plot_timing_graph([flex_attention_fwd_time, flex_attention_bwd_time, "FlexAttention"], 
                [xformers_sdpa_with_mask_fwd_time, xformers_sdpa_with_mask_bwd_time, "xFormers/SDPA + mask"], 
                [fa_fwd_time, fa_bwd_time, "FA (Causal)"], seq_length=S,
                name=name, path=Path(f"plots/{name}/timing.png"))

# Plot mask
B = 1
H = 1
S = 32
D = 8
DOCUMENT_IDXS = torch.zeros(S, dtype=torch.int, device='cuda')
DOCUMENT_IDXS[:4] = 0
DOCUMENT_IDXS[4:8] = 1
for i in range(8, S, 4):
    DOCUMENT_IDXS[i : i + 4] = i // 4

q, k = [torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(2)]
mask_mod = multi_document_causal_mask

visualize_attention_scores(q, k, mask_mod=mask_mod, name=name, path=Path(f"plots/{name}/mask.png"))