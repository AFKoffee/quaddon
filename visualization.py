import matplotlib.pyplot as plt
import torch

def plot_quantiles(stats, sigma):
    plt.figure()

    plt.ylim(-20*sigma, 20*sigma)

    x = torch.arange(0, len(stats["1p"][0]))
    plt.plot(x, stats["min"][0], color="blue")
    plt.plot(x, stats["max"][0], color="blue")
    plt.plot(x, stats["1p"][0], color="red")
    plt.plot(x, stats["99p"][0], color="red")
    plt.plot(x, stats["25p"][0], color="orange")
    plt.plot(x, stats["75p"][0], color="orange")
