## Quaddon
import numpy as np
import hadamard as hd
import torch

# 1. Create a random matrix
rows = 256
cols = 256
W_orig = torch.rand((rows, cols)) * 20 - 10

# 2. Transform the weights
W_had = hd.matmul_hadU(W_orig, transpose=False)

# 3. Calculate the µ-incoherance and visualize outlier distribution
mu_inc_orig = None
mu_inc_had = None
# ---> Create cool plot here ...

# 4. Quantize the weights
W_orig_q = None
W_had_q = None

# 5. Dequantize the weights
W_orig_d = None
W_had_d = None

# 6. Revert the transformation
W_had_r = hd.matmul_hadU(W_had_d, transpose=True)

# 7. Compare the difference to the original matrix
diff_orig = torch.sum(torch.abs(W_orig - W_orig_d))
diff_had = torch.sum(torch.abs(W_orig - W_had_r))