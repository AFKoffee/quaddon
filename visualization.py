import matplotlib.pyplot as plt
from utils import WeightDistribution, Results
import numpy as np

def plot_quantiles(analysis: WeightDistribution, y_limit):
    plt.figure()

    plt.ylim(-y_limit, y_limit)

    x = np.arange(0, analysis.get_nfeatures())
    plt.plot(x, analysis.min, color="blue")
    plt.plot(x, analysis.max, color="blue")
    plt.plot(x, analysis.q1, color="red")
    plt.plot(x, analysis.q99, color="red")
    plt.plot(x, analysis.q25, color="orange")
    plt.plot(x, analysis.q75, color="orange")


def plot_incoherence(results: Results):
    inc_orig = results.get_orig_incoherence()
    inc_had = results.get_had_incoherence()

    assert len(inc_orig) == len(inc_had)
    x = np.arange(0, len(inc_orig))

    plt.figure()
    plt.plot(x, inc_orig, color="blue")
    plt.plot(x, inc_had, color="orange")


def plot_mae(results: Results):
    mae_orig = results.get_orig_mae()
    mae_had = results.get_had_mae()

    assert len(mae_orig) == len(mae_had)
    x = np.arange(0, len(mae_orig))

    plt.figure()
    plt.plot(x, mae_orig, color="blue")
    plt.plot(x, mae_had, color="orange")


def scatter_inc_mae(results: Results):
    inc_orig = results.get_orig_incoherence()
    mae_orig = results.get_orig_mae()

    inc_had = results.get_had_incoherence()
    mae_had = results.get_had_mae()

    assert len(inc_orig) == len(inc_had)
    assert len(inc_had) == len(mae_orig)
    assert len(mae_orig) == len(mae_had)

    plt.figure()
    plt.scatter(inc_orig, mae_orig, color="blue")
    plt.scatter(inc_had, mae_had, color="orange")