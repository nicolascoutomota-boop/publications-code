import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "results/sim_t"
estr_cols = ['ple', 'plg', 'plw', 'pls', 'pld']
estr_colors = ['#a559aa', '#59a89c', '#f0c571', '#e02b35', '#082a54']
estr_label = ['Intercept', 'Linear', 'Log-linear', 'Spline', 'Disjoint']
sample_sizes = [250, 500, 1000, 2000]
times = [5, 10, 15, 20, 25, 30]

fig, axs = plt.subplots(4, 1, figsize=(7, 8))

for n, ax in zip(sample_sizes, axs):
    for estr, color, label in zip(estr_cols, estr_colors, estr_label):
        bias = []
        for t in times:
            d = pd.read_csv(file_name + str(t) + "_n" + str(n) + ".csv")
            bias_t = np.nanmean(d[estr + '_b'])
            bias.append(bias_t)
        ax.axhline(0, color='gray', linestyle=':')
        ax.plot(times, bias, '-o', color=color, label=label)
        ax.set_title(r"$n="+str(n)+r"$")
        ax.set_ylim([-0.1, 0.1])
        ax.set_xlim([-2, 32])
        ax.set_xticks([0, ] + times)
        ax.set_xticklabels(["", "", "", "", "", "", ""])


axs[0].legend(fontsize=8)
axs[-1].set_xticklabels(["0", "5", "10", "15", "20", "25", "30"])
axs[-1].set_xlabel("Time")
fig.supylabel('Bias')
plt.tight_layout()
plt.savefig("appendix_bias_plot.png", format='png', dpi=300)
plt.show()


fig, axs = plt.subplots(4, 1, figsize=(7, 8))

for n, ax in zip(sample_sizes, axs):
    for estr, color, label in zip(estr_cols, estr_colors, estr_label):
        bias = []
        for t in times:
            d = pd.read_csv(file_name + str(t) + "_n" + str(n) + ".csv")
            bias_t = np.nanstd(d[estr + '_b'])
            bias.append(bias_t)
        ax.plot(times, bias, '-o', color=color, label=label)
        ax.set_title(r"$n="+str(n)+r"$")
        ax.set_ylim([0, 0.09])
        ax.set_xlim([-2, 32])
        ax.set_xticks([0, ] + times)
        ax.set_xticklabels(["", "", "", "", "", "", ""])


axs[0].legend(fontsize=8)
axs[-1].set_xticklabels(["0", "5", "10", "15", "20", "25", "30"])
axs[-1].set_xlabel("Time")
fig.supylabel('Empirical Standard Error')
plt.tight_layout()
plt.savefig("appendix_ese_plot.png", format='png', dpi=300)
plt.show()


fig, axs = plt.subplots(4, 1, figsize=(7, 8))

for n, ax in zip(sample_sizes, axs):
    for estr, color, label in zip(estr_cols, estr_colors, estr_label):
        bias = []
        for t in times:
            d = pd.read_csv(file_name + str(t) + "_n" + str(n) + ".csv")
            bias_t = np.nanmean(d[estr + '_c'])
            bias.append(bias_t)
        ax.axhline(0.95, color='gray', linestyle=':')
        ax.plot(times, bias, '-o', color=color, label=label)
        ax.set_title(r"$n="+str(n)+r"$")
        ax.set_ylim([0, 1])
        ax.set_xlim([-2, 32])
        ax.set_xticks([0, ] + times)
        ax.set_xticklabels(["", "", "", "", "", "", ""])


axs[0].legend(fontsize=8)
axs[-1].set_xticklabels(["0", "5", "10", "15", "20", "25", "30"])
axs[-1].set_xlabel("Time")
fig.supylabel('95% Confidence Interval Coverage')
plt.tight_layout()
plt.savefig("appendix_cover_plot.png", format='png', dpi=300)
plt.show()
