import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

jet = cm = plt.get_cmap('Accent') 
cNorm  = colors.Normalize(vmin=0, vmax=2)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

sig = lambda x: 1 / (1 + np.exp(-1 * x))
log_sig = lambda x: np.log(sig(x))
H = lambda alpha : -1 * alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)
H_x = lambda alpha, x: alpha * x - H(alpha)


X = np.arange(-10, 10, 0.1)

sig_x = [sig(x) for x in X]
log_sig_x = [log_sig(x) for x in X]



for i, a in enumerate([0.000000000001, 0.5, 0.9999999999]):
    H_xs = [H_x(a, x) for x in X]
    plt.plot(X, H_xs, label=r"$\alpha = {0}$".format(a), \
        color=scalarMap.to_rgba(i), alpha=0.8)

plt.plot(X, log_sig_x, label=r"$\log\sigma(x)$", color='red')
plt.xlabel(r"$x$")

plt.legend()
plt.savefig("Figures/bounds.pdf", format="pdf")
plt.show()

plt.plot(np.arange(0.0000001, 0.999999999, 0.01), [H(a) for a in np.arange(0, 1, 0.01)],\
    label=r"$H(\alpha)$")
plt.ylabel(r"$H(\alpha)$")
plt.xlabel(r"$x$")
plt.legend()
plt.savefig("Figures/H_x.pdf", format="pdf")
