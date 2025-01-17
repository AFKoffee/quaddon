import numpy as np
from tqdm import tqdm
from hadamard import random_hadamard_matrix
from quantization import flexible_quant, flexible_dequant
import torch
from utils import Config, WeightDistribution, Results
from visualization import plot_quantiles
import matplotlib.pyplot as plt

def mu_incoherence(W):
    # The last two dimensions are expected to be the rows and cols of the matrix
    rows = W.shape[-2]
    cols = W.shape[-1]
    return (torch.max(torch.abs(W)) / torch.norm(W, p="fro")) * np.sqrt(rows * cols) 

def analyze_matrix(matrix, results: WeightDistribution):
    m_min = torch.min(matrix, dim=-1).values
    m_quantiles = torch.quantile(
        matrix, 
        torch.tensor([0.01, 0.25, 0.75, 0.99], dtype=torch.float64), 
        dim=-1,
        interpolation="linear"
    )
    m_max = torch.max(matrix, dim=-1).values

    results.fill(
        min=m_min.tolist(),
        p1=m_quantiles[0].tolist(),
        p25=m_quantiles[1].tolist(),
        p75=m_quantiles[2].tolist(),
        p99=m_quantiles[3].tolist(),
        max=m_max.tolist()
    )

def process_sample(W_orig, results: Results, layer_idx: int, config: Config):
    # 2. Transform the weights
    Q1 = random_hadamard_matrix(W_orig.shape[-2])
    Q2 = random_hadamard_matrix(W_orig.shape[-1])
    W_had = Q1.T @ W_orig @ Q2

    if config.validate:
        assert torch.allclose(Q1 @ Q1.T, torch.eye(Q1.shape[-1], dtype=torch.float64))
        assert torch.allclose(Q2 @ Q2.T, torch.eye(Q2.shape[-1], dtype=torch.float64))
        assert torch.allclose(Q1 @ W_had @ Q2.T, W_orig)

    if config.quantile_analysis:
        orig_dist = WeightDistribution(layer_idx, False)
        analyze_matrix(W_orig, orig_dist)

        had_dist = WeightDistribution(layer_idx, True)
        analyze_matrix(W_had, had_dist)

        y_limit = max(orig_dist.find_abs_max(), had_dist.find_abs_max())
        plot_quantiles(orig_dist, y_limit, config)
        plot_quantiles(had_dist, y_limit, config)

        if config.display:
            plt.show()

    # 3. Calculate the µ-incoherance
    mu_inc_orig = mu_incoherence(W_orig)
    mu_inc_had = mu_incoherence(W_had)

    results.push_incoherence(mu_inc_orig.item(), mu_inc_had.item())

    # 4. Quantize the weights
    W_orig_q, s_orig, z_orig = flexible_quant(W_orig, config.bitwidth)
    W_had_q, s_had, z_had = flexible_quant(W_had, config.bitwidth)

    # 5. Dequantize the weights
    W_orig_d = flexible_dequant(W_orig_q, s_orig, z_orig)
    W_had_d = flexible_dequant(W_had_q, s_had, z_had)

    # 6. Revert the transformation
    W_had_r = Q1 @ W_had_d @ Q2.T

    # 7. Compare the difference to the original matrix
    N = W_orig.numel()
    diff_orig = torch.sum(torch.abs(W_orig - W_orig_d)) / N
    diff_had = torch.sum(torch.abs(W_orig - W_had_r)) / N

    results.push_mae(diff_orig.item(), diff_had.item())

def analyze_model(model, config: Config):
    results = Results()

    for (idx, block) in enumerate(tqdm(model.layers)):
        weights = block.mlp.down_proj.weight
        process_sample(weights.to(dtype=torch.float64), results, idx, config)
        
        # Close all figures before next loop iteration
        plt.close('all')

    return results
