## Quaddon
import argparse
import torch
from analysis import analyze_model
from utils import Config, EnhancedJSONEncoder
from visualization import plot_incoherence, plot_mae, scatter_inc_mae
import matplotlib.pyplot as plt
import transformers
from pathlib import Path
import json
import os

def main():
    # Setup the cli
    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=60)
    parser = argparse.ArgumentParser(formatter_class=formatter)

    # Execution options
    parser.add_argument("-b", "--bitwidth", default=4, help="the quantization bitwidth")
    parser.add_argument("-d", "--display", action="store_true", help="display the created plots during execution")
    parser.add_argument("-s", "--save", action="store_true", help="save the created plots to file")
    parser.add_argument("-q", "--quantile-analysis", action="store_true", help="perform a quantile analysis of the weights")
    parser.add_argument("--validate", action="store_true", help="include validation checks")
    parser.add_argument("-o", "--out-dir", default="out", help="output directory")
    parser.add_argument("-m", "--module", default="down-proj", choices=["down-proj", "qkv-proj"], help="the linear module to analyze per transformer block")

    # Seed option for deteministic randomized hadamard matrices
    parser.add_argument("--seed", default=None, help="set a seed for deterministic randomized hadamard matrices")

    # Run the actual demo
    args = parser.parse_args()


    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed = args.seed
    else:
        seed = torch.seed()

    config = Config(
        args.bitwidth, 
        args.display, 
        args.save, 
        args.quantile_analysis, 
        args.validate, 
        args.out_dir,
        args.module,
        seed
    )

    if config.save:
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.sep.join([config.out_dir, "config.json"]), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map = "cpu",
        torch_dtype="auto",
        trust_remote_code=True
    )

    results = analyze_model(llm.model, config)
    
    plot_incoherence(results, config)
    plot_mae(results, config)
    scatter_inc_mae(results, config)
    
    if config.display:
        plt.show()


if __name__ == "__main__":
    main()