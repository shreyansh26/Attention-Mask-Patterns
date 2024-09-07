from torch.nn.attention.flex_attention import create_block_mask

S = 2048
SLIDING_WINDOW = 512

def bidirectional_sliding_window(b, h, q_idx, kv_idx):
    windowed_mask = (q_idx - kv_idx).abs() <= SLIDING_WINDOW
    return windowed_mask

block_mask = create_block_mask(bidirectional_sliding_window, B=None, H=None, Q_LEN=S, KV_LEN=S)

print(block_mask)