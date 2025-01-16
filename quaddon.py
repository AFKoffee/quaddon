## Quaddon
import argparse
import torch
from comparison import analyze_model
from visualization import plot_incoherence, plot_mae, scatter_inc_mae
import matplotlib.pyplot as plt
import transformers
import json

def main():
    # Setup the cli
    parser = argparse.ArgumentParser()

    # Seed option for deteministic randomized hadamard matrices
    parser.add_argument("--seed", default=None)

    # Run the actual demo
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map = "cpu",
        torch_dtype="auto",
        trust_remote_code=True
    )

    results = analyze_model(llm.model)
    
    plot_incoherence(results)
    plot_mae(results)
    scatter_inc_mae(results)
    plt.show()


if __name__ == "__main__":
    main()