import torch
from torch.nn.attention.flex_attention import create_block_mask, or_masks

from functools import partial

S = 2048
SLIDING_WINDOW = 512
TOK_IDXS_FOR_GLOBAL_ATTN = [0, 512, 1024]
BIDIRECTIONAL_COMP_IDXS = torch.ones(S, device='cuda') * -1000

for idx in TOK_IDXS_FOR_GLOBAL_ATTN:
    BIDIRECTIONAL_COMP_IDXS[idx] = idx

def bidirectional_sliding_window(b, h, q_idx, kv_idx):
    windowed_mask = (q_idx - kv_idx).abs() <= SLIDING_WINDOW
    return windowed_mask

def global_attention_mask(b, h, q_idx, kv_idx, bidirectional_comp_idxs):
    global_attn_mask = (q_idx == bidirectional_comp_idxs[q_idx]) | (kv_idx == bidirectional_comp_idxs[kv_idx])
    return global_attn_mask

block_mask = create_block_mask(or_masks(bidirectional_sliding_window, partial(global_attention_mask, bidirectional_comp_idxs=BIDIRECTIONAL_COMP_IDXS)), B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
print(block_mask)