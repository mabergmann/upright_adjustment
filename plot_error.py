import matplotlib.pyplot as plt
import numpy as np


with open("out.txt", "r") as f:
    lines = f.readlines()
x = [float(x) for x in lines]
y = np.asarray(range(1, len(x) + 1)) * 100 / len(x)
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(8, 2))
plt.xlim(0, 15)
plt.plot(x, y)
plt.xlabel("Error (degrees)")
plt.ylabel("Percentile")
plt.savefig("upright_percentileVersusError", bbox_inches="tight")
