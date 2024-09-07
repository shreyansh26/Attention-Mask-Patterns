import torch
import matplotlib.pyplot as plt
from torch.nn.attention.flex_attention import create_block_mask, and_masks, create_mask

SLIDING_WINDOW = 256
DILATION_FACTOR = 2
S = 2048
def bidirectional_sliding_window(b, h, q_idx, kv_idx):
    windowed_mask = (q_idx - kv_idx).abs() <= DILATION_FACTOR * SLIDING_WINDOW
    return windowed_mask

def dilated_mask(b, h, q_idx, kv_idx):
    dilated_mask = torch.eq(torch.fmod((q_idx - kv_idx).abs(), DILATION_FACTOR), 0)
    return dilated_mask

block_mask = create_block_mask(and_masks(bidirectional_sliding_window, dilated_mask), B=None, H=None, Q_LEN=S, KV_LEN=S)

print(block_mask)

mask = create_mask(and_masks(bidirectional_sliding_window, dilated_mask), 1, 1, S, S).double()

mask = mask.squeeze()
plt.imshow(mask.cpu().numpy(), cmap='hot', interpolation='nearest')
plt.savefig('foo.png')
