import torch
from triton.testing import do_bench
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, create_mask

import matplotlib.pyplot as plt
from pathlib import Path
from utils import visualize_attention_scores

torch.set_default_device('cuda')

B = 8
H = 16
S = 2048
D = 64

q, k, v = [torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float16) for _ in range(3)]

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal, B=B, H=None, Q_LEN=S, KV_LEN=S)
mask = create_mask(causal, B=1, H=1, Q_LEN=S, KV_LEN=S)

print("Causal Mask:", block_mask[0])

# Benchmark and Correctness
flex_attention = torch.compile(flex_attention)

print("Flex Attention:", flex_attention(q, k, v, block_mask=block_mask).sum())
print("Flex Attention fwd:", do_bench(lambda: flex_attention(q, k, v, block_mask=block_mask).sum()))
print("Flex Attention bwd:", do_bench(lambda: flex_attention(q, k, v, block_mask=block_mask).sum().backward()))

print("xformers/sdpa with mask:", scaled_dot_product_attention(q, k, v, attn_mask=mask).sum())
print("xformers/sdpa with mask fwd:", do_bench(lambda: scaled_dot_product_attention(q, k, v, attn_mask=mask).sum()))
print("xformers/sdpa with mask bwd:", do_bench(lambda: scaled_dot_product_attention(q, k, v, attn_mask=mask).sum().backward()))

# Only enable flash attention backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    print("FA (causal):", scaled_dot_product_attention(q, k, v, is_causal=True).sum())
    print("FA (causal) fwd:", do_bench(lambda: scaled_dot_product_attention(q, k, v, is_causal=True).sum()))
    print("FA (causal) bwd:", do_bench(lambda: scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()))


# Plot
B = 1
H = 1
S = 32
D = 8
q, k = [torch.ones(B, H, S, D, dtype=torch.float16) for _ in range(2)]
visualize_attention_scores(q, k, mask_mod=causal, name="causal", path=Path("images/causal.png"))