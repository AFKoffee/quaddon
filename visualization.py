import matplotlib.pyplot as plt
from utils import WeightDistribution, Results, Config
import numpy as np
import os

def plot_quantiles(analysis: WeightDistribution, y_limit: float, config: Config):
    plt.figure()

    plt.ylim(-y_limit, y_limit)

    x = np.arange(0, analysis.get_nfeatures(config.validate))
    plt.plot(x, analysis.min, color="blue", label="min/max")
    plt.plot(x, analysis.max, color="blue", label="min/max")
    plt.plot(x, analysis.p1, color="red", label="1p/99p")
    plt.plot(x, analysis.p99, color="red", label="1p/99p")
    plt.plot(x, analysis.p25, color="orange", label="25p/75p")
    plt.plot(x, analysis.p75, color="orange", label="25p/75p")

    # Remove duplicate lables
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    
    plt.legend(newHandles, newLabels, loc="upper right")
    plt.title(
        "Weight Distribution of the Weights in Layer {idx} ({mode})"
            .format(idx=analysis.layer_idx, mode="hadamard" if analysis.is_transformed else "original")
    )
    plt.xlabel("Hidden Dimension Index")
    plt.ylabel("Weight Value")

    if config.save:
        plt.savefig(os.sep.join(
            [config.out_dir, "w_dist_{idx}_{mode}.png".format(idx=analysis.layer_idx, mode="hadamard" if analysis.is_transformed else "original")]
        ))


def plot_incoherence(results: Results, config: Config):
    inc_orig = results.get_orig_incoherence()
    inc_had = results.get_had_incoherence()

    if config.validate:
        assert len(inc_orig) == len(inc_had)
    
    x = np.arange(0, len(inc_orig))

    plt.figure()
    plt.plot(x, inc_orig, color="blue", label="original weights")
    plt.plot(x, inc_had, color="orange", label="hadamard weights")

    plt.legend(loc="upper left")
    plt.title("µ-incoherences across Transformer Blocks")
    plt.xlabel("Block Idx")
    plt.ylabel("µ-incoherence")

    if config.save:
        plt.savefig(os.sep.join(
            [config.out_dir, "incoherence.png"]
        ))


def plot_mae(results: Results, config: Config):
    mae_orig = results.get_orig_mae()
    mae_had = results.get_had_mae()

    if config.validate:
        assert len(mae_orig) == len(mae_had)
    
    x = np.arange(0, len(mae_orig))

    plt.figure()
    plt.plot(x, mae_orig, color="blue", label="original weights")
    plt.plot(x, mae_had, color="orange", label="hadamard weights")

    plt.legend(loc="upper left")
    plt.title("Quantization Errors across Transformer Blocks")
    plt.xlabel("Block Idx")
    plt.ylabel("Mean-Absolute-Error (MAE)")

    if config.save:
        plt.savefig(os.sep.join(
            [config.out_dir, "mae.png"]
        ))


def scatter_inc_mae(results: Results, config: Config):
    inc_orig = results.get_orig_incoherence()
    mae_orig = results.get_orig_mae()

    inc_had = results.get_had_incoherence()
    mae_had = results.get_had_mae()

    if config.validate:
        assert len(inc_orig) == len(inc_had)
        assert len(inc_had) == len(mae_orig)
        assert len(mae_orig) == len(mae_had)

    plt.figure()
    plt.scatter(inc_orig, mae_orig, color="blue", label="original weights")
    plt.scatter(inc_had, mae_had, color="orange", label="hadamard weights")

    plt.legend(loc="upper left")
    plt.title("µ-incoherence vs. Mean-Absolute-Error per Transformer Block")
    plt.xlabel("µ-incoherence")
    plt.ylabel("Mean-Absolute-Error (MAE)")

    if config.save:
        plt.savefig(os.sep.join(
            [config.out_dir, "inc_vs_mae.png"]
        ))