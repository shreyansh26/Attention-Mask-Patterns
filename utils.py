'''
From - https://github.com/pytorch-labs/attention-gym/blob/75867424a1d4391bff49527029d3612a09dd67e2/attn_gym/utils.py
'''

import torch
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
)
from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from contextlib import nullcontext

Tensor = torch.Tensor


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature],
    mask_mod: Optional[_mask_mod_signature],
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        if type == _ModificationType.SCORE_MOD:
            out = mod(scores, b, h, m, n)
        else:
            out = mod(b, h, m, n)

    return out


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    Returns:
        None
    """
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide either score_mod or mask_mod"
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Mask visualization saved as {file_path}")

def plot_timing_graph(
    flex_attention_timings: list,
    xformers_sdpa_with_mask_timings: list,
    fa_timings: list,
    seq_length: int,
    name: str = "attention_timings",
    path: Optional[Path] = None,
):
    """
    Plot a bar graph comparing fwd and bwd timings for different kernels
    """
    fwd_times = [flex_attention_timings[0], xformers_sdpa_with_mask_timings[0], fa_timings[0]]
    bwd_times = [flex_attention_timings[1], xformers_sdpa_with_mask_timings[1], fa_timings[1]]
    labels = [flex_attention_timings[2], xformers_sdpa_with_mask_timings[2], fa_timings[2]]
    
    # Set up the bar positions
    x = np.arange(len(labels))  # Kernel indices
    width = 0.35  # Width of the bars
    
    # Create subplots for forward and backward pass
    fig, ax = plt.subplots()
    
    # Plot bars for forward and backward times
    bars_fwd = ax.bar(x - width/2, fwd_times, width, label='Forward Pass', color='b')
    bars_bwd = ax.bar(x + width/2, bwd_times, width, label='Backward Pass', color='g')
    
    title = _name_to_title(name)
    ax.axhline(fwd_times[0], linestyle='--', color='r', linewidth=0.7)
    ax.axhline(bwd_times[0], linestyle='--', color='r', linewidth=0.7)
    # Adding labels and title
    ax.set_xlabel('Kernels')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(f"{title}; S={seq_length}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Adding the execution time values on top of each bar
    for bar in bars_fwd:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    for bar in bars_bwd:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    # Display the plot
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Benchmark visualization saved as {file_path}")
