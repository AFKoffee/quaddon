import numpy as np
from hadamard import random_hadamard_matrix
from quantization import flexible_quant, flexible_dequant
import torch

def mu_incoherence(W):
    # The last two dimensions are expected to be the rows and cols of the matrix
    rows = W.shape[-2]
    cols = W.shape[-1]
    return (torch.max(torch.abs(W)) / torch.norm(W, p="fro")) * np.sqrt(rows * cols) 

def analyze_matrix(matrix, is_transfomred, results):
    # TODO
    pass

def process_sample(W_orig, results):
    # 2. Transform the weights
    Q1 = random_hadamard_matrix(W_orig.shape[-2])
    assert torch.allclose(Q1 @ Q1.T, torch.eye(Q1.shape[-1], dtype=torch.float64))

    Q2 = random_hadamard_matrix(W_orig.shape[-1])
    assert torch.allclose(Q2 @ Q2.T, torch.eye(Q2.shape[-1], dtype=torch.float64))
    
    W_had = Q1.T @ W_orig# @ Q2
    #assert torch.allclose(Q1 @ W_had @ Q2.T, W_orig)

    analyze_matrix(W_orig, False, results)
    analyze_matrix(W_had, True, results)

    # 3. Calculate the µ-incoherance
    mu_inc_orig = mu_incoherence(W_orig)
    mu_inc_had = mu_incoherence(W_had)

    results["incoherence"]["original"].append(mu_inc_orig.item())
    results["incoherence"]["hadamard"].append(mu_inc_had.item())

    # 4. Quantize the weights
    W_orig_q, s_orig, z_orig = flexible_quant(W_orig, 6)
    W_had_q, s_had, z_had = flexible_quant(W_had, 6)

    # 5. Dequantize the weights
    W_orig_d = flexible_dequant(W_orig_q, s_orig, z_orig)
    W_had_d = flexible_dequant(W_had_q, s_had, z_had)

    # 6. Revert the transformation
    W_had_r = Q1 @ W_had_d #@ Q2.T

    # 7. Compare the difference to the original matrix
    N = W_orig.numel()
    diff_orig = torch.sum(torch.abs(W_orig - W_orig_d)) / N
    diff_had = torch.sum(torch.abs(W_orig - W_had_r)) / N

    results["mean-absolute-error"]["original"].append(diff_orig.item())
    results["mean-absolute-error"]["hadamard"].append(diff_had.item())
    

def perform_comparison(nsamples, rows, cols, distribution):
    results = {
        "incoherence": {
            "original": [],
            "hadamard": []
        },
        "mean-absolute-error": {
            "original": [],
            "hadamard": []
        },
        "stats": {
            "original": {
                "min": [],
                "1p": [],
                "25p": [],
                "75p": [],
                "99p": [],
                "max": [],
            },
            "hadamard": {
                "min": [],
                "1p": [],
                "25p": [],
                "75p": [],
                "99p": [],
                "max": [],
            }
        }
    }

    for _ in range(nsamples):
        # Sample a random weight matrix according to given
        W_orig = distribution.sample((rows, cols))

        # Introduce some outliers
        mask = torch.rand_like(W_orig) < 0.1
        W_orig[mask] += 0.5

        # Process the matrix
        process_sample(W_orig.to(dtype=torch.float64), results)
    
    return results
