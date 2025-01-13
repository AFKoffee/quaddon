import numpy as np
from hadamard import matmul_hadU
from quantization import four_bit_quant, four_bit_dequant
import torch

def mu_incoherence(W):
    # The last two dimensions are expected to be the rows and cols of the matrix
    rows = W.shape[-2]
    cols = W.shape[-1]
    return (torch.max(W) / torch.norm(W, p="fro")) * np.sqrt(rows * cols) 

def analyze_matrix(matrix, is_transfomred, results):
    # TODO
    pass

def process_sample(W_orig, results):
    # 2. Transform the weights
    W_had = matmul_hadU(W_orig, transpose=False)

    analyze_matrix(W_orig, False, results)
    analyze_matrix(W_had, True, results)

    # 3. Calculate the µ-incoherance
    mu_inc_orig = mu_incoherence(W_orig)
    mu_inc_had = mu_incoherence(W_had)

    results["incoherence"]["original"].append(mu_inc_orig)
    results["incoherence"]["hadamard"].append(mu_inc_had)

    # 4. Quantize the weights
    W_orig_q, s_orig, z_orig = four_bit_quant(W_orig)
    W_had_q, s_had, z_had = four_bit_quant(W_had)

    # 5. Dequantize the weights
    W_orig_d = four_bit_dequant(W_orig_q, s_orig, z_orig)
    W_had_d = four_bit_dequant(W_had_q, s_had, z_had)

    # 6. Revert the transformation
    W_had_r = matmul_hadU(W_had_d, transpose=True)

    # 7. Compare the difference to the original matrix
    diff_orig = torch.sum(torch.abs(W_orig - W_orig_d))
    diff_had = torch.sum(torch.abs(W_orig - W_had_r))

    results["difference"]["original"].append(diff_orig)
    results["difference"]["hadamard"].append(diff_had)
    

def perform_comparison(nsamples, rows, cols, distribution):
    results = {
        "incoherence": {
            "original": [],
            "hadamard": []
        },
        "difference": {
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
        W_orig = distribution.sample((rows, cols)).unsqueeze(0)
        process_sample(W_orig, results)
    
    return results
