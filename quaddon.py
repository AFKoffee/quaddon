## Quaddon
import argparse
import torch
from comparison import perform_comparison

parser = argparse.ArgumentParser()

# Arguments for the matrix dimensions
parser.add_argument("-r", "--rows", default=256)
parser.add_argument("-c", "--cols", default=256)

# Arguments for the matrix initialization
parser.add_argument("-d", "--distribution", choices=["uniform", "gaussian"], default="uniform")
parser.add_argument("-m", "--mean", default=0.0)
parser.add_argument("-s", "--sigma", default=10.0)

# Arguments for matrix quantization
# parser.add_argument(...)

# Arguments for the sampling process
parser.add_argument("--seed", default=None)
parser.add_argument("-n", "--nsamples", default=5)

# Helper functions
def get_distribution(name, mean, sigma):
    if name == "uniform":
        low = mean - (sigma/2)
        high = mean + (sigma/2)
        return torch.distributions.Uniform(low, high)
    elif name == "gaussian":
        return torch.distributions.Normal(mean, sigma)
    else:
        raise Exception("Error: Unknown distribution name given!")

def setup_rng(dist_name, mean, sigma, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    return get_distribution(dist_name, mean, sigma)

# Run the actual demo
args = parser.parse_args()
distribution = setup_rng(args.distribution, args.mean, args.sigma, args.seed)
results = perform_comparison(args.nsamples, args.rows, args.cols, distribution)

print(results)