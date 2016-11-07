import numpy as np
from numpy.testing import assert_array_equal

from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import NSlackSSVM, SubgradientMBestSSVM
from pystruct.datasets import generate_blocks_multinomial
from pystruct.utils import make_grid_edges
from pystruct.utils import SaveLogger


def edge_list_to_features(edge_list):
    edges = np.vstack(edge_list)
    edge_features = np.zeros((edges.shape[0], 2))
    edge_features[:len(edge_list[0]), 0] = 1
    edge_features[len(edge_list[0]):, 1] = 1
    return edge_features

def pickBest(ypreds,ytrue):
	best = (None, np.inf)
	for i, ypred in enumerate(ypreds):
		loss = np.sum(ypred!=ytrue)
		if loss<best[1]:
			best = (i, loss)
	return ypreds[best[0]]

# dataset and model
N = 20
noise = 3.0
X_, Y_ = generate_blocks_multinomial(n_samples=N, noise=noise, seed=0)
G = [make_grid_edges(x, return_lists=True) for x in X_]
edge_features = [edge_list_to_features(edge_list) for edge_list in G]
edges = [np.vstack(g) for g in G]
X = zip([x.reshape(-1, 3) for x in X_], edges, edge_features)
Y = [y.ravel() for y in Y_]
Xtrain = np.array(X[0:int(np.floor(N/2))])
Ytrain = np.array(Y[0:int(np.floor(N/2))])
Xtest = np.array(X[int(np.floor(N/2)):])
Ytest = np.array(Y[int(np.floor(N/2)):])

crf = EdgeFeatureGraphCRF(n_states=3, n_edge_features=2)


## SSVM
ssvm_logger = './logExampleSSVM.pickle'
ssvm = NSlackSSVM(model=crf, max_iter=10, C=1, check_constraints=False, 
			logger=SaveLogger(ssvm_logger,save_every=100) )
ssvm.fit(Xtrain, Ytrain)
Ypred_ssvm = ssvm.predict(Xtest)


## CSSVM
M = 5           # number of models in ensemble
gamma = 0.001   # diversity weight
C = 1           # regularizer weight
cssvm = SubgradientMBestSSVM(crf, verbose=1, C=C, max_iter=100, n_jobs=1,
						M=M, gamma=gamma, initialize_w_by=ssvm_logger, 
						momentum=0.0, learning_rate='auto', break_on_no_constraints=True, batch_size=None,
						decay_t0=1, decay_exponent=0.5, averaging=None, shuffle=False, sample_assignment_strategy='all',
						online=False, break_on_no_loss_improvement=2,parallel_inference=True,
						zero_constraint=None, force_moment=1)
cssvm.fit(Xtrain, Ytrain)
Ypreds_cssvm = cssvm.predict(Xtest)


## evaluation
print 'Hamming loss:'
print 'SSVM:  ', np.mean([np.sum(yp!=yt) for (yp, yt) in zip(Ypred_ssvm,Ytest)])
print 'CSSVM: ', np.mean([np.sum(pickBest(yps,yt)!=yt) for (yps, yt) in zip(Ypreds_cssvm,Ytest)])
