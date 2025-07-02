import matplotlib.pyplot as plt
import numpy as np
import torch


def get_true_mask(X1_range, X2_range, y, noise, lb, ub):
    X1, X2 = np.meshgrid(X1_range, X2_range)
    Y_mean_grid = y.reshape(X1.shape)
    Y_noise_grid = noise.reshape(X1.shape) # noise is std
    # Y_noise_grid = torch.sqrt(pred_noise.reshape(X1.shape)) # noise is var

    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # cp1 = axes[0].contourf(X1, X2, Y_mean_grid, cmap="Blues", levels=20)
    # fig.colorbar(cp1, ax=axes[0])
    true_mask = (Y_mean_grid > lb) * (Y_mean_grid < ub)
    
    # axes[0].scatter(X1[true_mask], X2[true_mask], color='#fed976', marker='x', alpha=0.5)
    # axes[0].set_xlabel("x1")
    # axes[0].set_ylabel("x2")
    # axes[0].set_title("Rg Mean")
    
    # cp2 = axes[1].contourf(X1, X2, Y_noise_grid, cmap="YlOrBr", levels=20)
    # fig.colorbar(cp2, ax=axes[1])
    # axes[1].set_xlabel("x1")
    # axes[1].set_ylabel("x2")
    # axes[1].set_title("Rg Noise")
    
    # plt.tight_layout()
    # plt.show()
    
    return true_mask


def plot_F1(F1_list):
    iters = np.arange(np.asarray(F1_list).shape[0])
    plt.figure(figsize=(12, 8))
    plt.plot(iters, F1_list, marker='o', linestyle='-', linewidth=3, markersize=12, markeredgewidth=2)
    plt.xlabel("No. of Iterations", fontsize=36)
    plt.ylabel("F1 Score", fontsize=36)
    plt.xticks(iters, fontsize=20)
    plt.yticks(fontsize=28)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()