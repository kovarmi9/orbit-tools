#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Input file: first argument, or diff.txt in the current directory.
# (Path(__file__) cannot be used: in a PyInstaller --onefile exe it points to a temp dir.)
diff_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diff.txt")

data = np.loadtxt(diff_file, usecols=(0, 1, 2, 3),
                  converters={0: lambda s: __import__("datetime").datetime.fromisoformat(s).timestamp()},
                  encoding="utf-8", comments=None, skiprows=1)

t = (data[:, 0] - data[0, 0]) / 3600.0
d = data[:, 1:] * 1000.0  # m → mm

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, sharey=True)
fig.suptitle("Rozdíly orbit", fontsize=15, fontweight="bold")

ylim = (-np.max(np.abs(d)) * 1.05, np.max(np.abs(d)) * 1.05)

for ax, col, label in zip(axes, d.T, ["Radiální směr (R)", "Transverzální směr (T)", "Normálový směr (N)"]):
    ax.axhline(0, color="black", linewidth=1.2, zorder=0)
    ax.plot(t, col, linewidth=1.0)
    ax.set_title(label, fontsize=11)
    ax.set_ylabel("Rozdíl [mm]")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Čas od začátku série [h]")
plt.tight_layout()
plt.show()
