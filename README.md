# Attention Mask Patterns

Using FlexAttention to compute attention with different masking patterns. 

The speedup over F.sdpa/xFormers and FA2 tends to increase with increasing sequence length. Timing plots are show for `seq_length=4096`.

### Causal mask
Mask             |  Exeecution Time
:-------------------------:|:-------------------------:
![](plots/causal/mask.png)  |  ![](plots/causal/timing.png)

### Causal sliding window mask
Mask             |  Exeecution Time
:-------------------------:|:-------------------------:
![](plots/causal_sliding_window/mask.png)  |  ![](plots/causal_sliding_window/timing.png)

### Bidirectional sliding window mask
Mask             |  Exeecution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_sliding_window/mask.png)  |  ![](plots/bidirectional_sliding_window/timing.png)

### Bidirectional dilated sliding window mask
Mask             |  Exeecution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_dilated_sliding_window/mask.png)  |  ![](plots/bidirectional_dilated_sliding_window/timing.png)

### Bidirectional global + local sliding window attention mask
Mask             |  Exeecution Time
:-------------------------:|:-------------------------:
![](plots/bidirectional_local_sliding_window_global_attention/mask.png)  |  ![](plots/bidirectional_local_sliding_window_global_attention/timing.png)


## Requirements
* Pytorch Nightly (for FlexAttention, to be released with Pytorch 2.5)
* Refer `requirements.txt` for other requirements