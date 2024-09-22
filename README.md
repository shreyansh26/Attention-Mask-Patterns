# Attention Mask Patterns

Using FlexAttention to compute attention with different masking patterns. 

The speedup over F.sdpa/xFormers and FA2 tends to increase with increasing sequence length. Timing plots are shown for `seq_length=4096`.

### Causal mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/causal/mask.png)  |  ![](plots/causal/timing.png)

### Causal sliding window mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/causal_sliding_window/mask.png)  |  ![](plots/causal_sliding_window/timing.png)

### Bidirectional sliding window mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_sliding_window/mask.png)  |  ![](plots/bidirectional_sliding_window/timing.png)

### Bidirectional dilated sliding window mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_dilated_sliding_window/mask.png)  |  ![](plots/bidirectional_dilated_sliding_window/timing.png)

### Bidirectional global + local sliding window attention mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_local_sliding_window_global_attention/mask.png)  |  ![](plots/bidirectional_local_sliding_window_global_attention/timing.png)

### PrefixLM mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/prefix_lm/mask.png)  |  ![](plots/prefix_lm/timing.png)

### Multi-document causal mask
Mask             |  Execution Time
:-------------------------:|:-------------------------:
![](plots/multi_document_causal_mask/mask.png)  |  ![](plots/multi_document_causal_mask/timing.png)


## Requirements
* Pytorch Nightly (for FlexAttention, to be released with Pytorch 2.5)
* Refer `requirements.txt` for other requirements