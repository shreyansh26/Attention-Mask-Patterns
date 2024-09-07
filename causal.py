from torch.nn.attention.flex_attention import create_block_mask

S = 2048

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S)

print(block_mask)