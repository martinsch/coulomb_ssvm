import numpy as np
from pystruct.datasets.synthetic_grids import generate_checker_multinomial, generate_blocks_multinomial
from pystruct import learners
from pystruct.plot_learning import plot_learning
from pystruct.inference.inference_methods import get_installed
import pystruct.models as crfs
import matplotlib.pyplot as plt

inference_method = get_installed(["qpbo", "ad3", "lp"])[0]

# experiment = 'checker'
experiment = 'blocks'

if experiment == 'checker':
    X, Y = generate_checker_multinomial(n_samples=10, noise=0.4)
elif experiment == 'blocks':
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.4, seed=1)
else:
    raise NotImplementedError

n_labels = len(np.unique(Y))
crf = crfs.GridCRF(n_states=n_labels, inference_method=inference_method)
clf_m1 = learners.SubgradientMBestSSVM(model=crf, verbose=1, max_iter=4000,
                                    M=1, gamma=0.1,
                                    sample_assignment_strategy='all',
                                    break_on_no_loss_improvement=2)
clf_m1.fit(X, Y)

M = 3
clf = learners.SubgradientMBestSSVM(model=crf, verbose=1, max_iter=1000,
                                    M=M, C=10, gamma=0.1,
                                    sample_assignment_strategy='all',
                                    break_on_no_loss_improvement=2)
clf.fit(X, Y)


if experiment == 'checker':
    Xtest, Ytest = generate_checker_multinomial(n_samples=20, noise=0.6)
elif experiment == 'blocks':
    Xtest, Ytest = generate_blocks_multinomial(n_samples=20, noise=0.6, seed=18)

Y_pred = clf.predict(Xtest)

plt.matshow(Ytest[0])
plt.title("Ground truth")

plt.matshow(clf_m1.predict(Xtest)[0][0].reshape(Ytest[0].shape))
plt.title("Prediction with M=1")

for m in xrange(M):
    plt.matshow(Y_pred[m][0].reshape(Ytest[0].shape))
    plt.title("Prediction %d" % m)

plot_learning(clf, time=False)
plt.show()