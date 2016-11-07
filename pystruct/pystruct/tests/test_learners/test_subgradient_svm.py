from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_less

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GridCRF, GraphCRF
from pystruct.learners import SubgradientSSVM
from pystruct.inference import get_installed
from pystruct.datasets import (generate_blocks_multinomial,
                               generate_checker_multinomial, generate_blocks)
from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
from pystruct.utils import SaveLogger


inference_method = get_installed(["qpbo", "ad3", "lp"])[0]


def test_multinomial_blocks_subgradient():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.6, seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks_subgradient_offline():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.6, seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=100, online=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_checker_subgradient():
    X, Y = generate_checker_multinomial(n_samples=10, noise=0.4)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient_parallel():
    # fixme: travis doesn't like parallelism?
    pass
    #testing subgradient ssvm on easy binary dataset
    #X, Y = generate_blocks(n_samples=10)
    #crf = GridCRF()
    #clf = SubgradientSSVM(model=crf, max_iter=100, C=1,
                          #momentum=.0, learning_rate=0.1, n_jobs=-1)
    #clf.fit(X, Y)
    #Y_pred = clf.predict(X)
    #assert_array_equal(Y, Y_pred)


def test_binary_blocks():
    #testing subgradient ssvm on easy binary dataset
    X, Y = generate_blocks(n_samples=5)
    crf = GridCRF(inference_method=inference_method)
    clf = SubgradientSSVM(model=crf)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_subgradient_svm_as_crf_pickling():

    iris = load_iris()
    X, y = iris.data, iris.target

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    Y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y, random_state=1)
    _, file_name = mkstemp()

    pbl = GraphCRF(n_features=4, n_states=3, inference_method='unary')
    logger = SaveLogger(file_name)
    svm = SubgradientSSVM(pbl, logger=logger, max_iter=100)
    svm.fit(X_train, y_train)

    assert_less(.97, svm.score(X_test, y_test))
    assert_less(.97, logger.load().score(X_test, y_test))


def test_multinomial_blocks_subgradient_batch():
    #testing cutting plane ssvm on easy multinomial dataset
    X, Y = generate_blocks_multinomial(n_samples=10, noise=0.6, seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, max_iter=100, batch_size=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    
    clf2 = SubgradientSSVM(model=crf, max_iter=100, batch_size=len(X))
    clf2.fit(X, Y)
    Y_pred2 = clf2.predict(X)
    assert_array_equal(Y, Y_pred2)


def test_binary_ssvm_attractive_potentials_edgefeaturegraph(inference_method="qpbo"):
    X, Y = generate_blocks(n_samples=10)
    crf = GridCRF(inference_method=inference_method)

    #######

    # convert X,Y to EdgeFeatureGraphCRF instances
    crf_edge = EdgeFeatureGraphCRF(inference_method=inference_method,
                                   symmetric_edge_features=[0]
                                    )
    X_edge = []
    Y_edge = []
    for i in range(X.shape[0]):
        unaries = X[i].reshape((-1, 2))
        edges = crf._get_edges(X[i])
        edge_feats = np.ones((edges.shape[0], 1))
        X_edge.append((unaries, edges, edge_feats))
        Y_edge.append((Y[i].reshape((-1,))))

    submodular_clf_edge = SubgradientSSVM(model=crf_edge, max_iter=100, C=1,
                                verbose=1,
                                zero_constraint=[4,7],
                                negativity_constraint=[5,6],
                                )

    # fit the model with non-negativity constraint on the off-diagonal potential
    submodular_clf_edge.fit(X_edge, Y_edge)

    assert submodular_clf_edge.w[5] == submodular_clf_edge.w[6] # symmetry constraint on edge features

    # # # bias doesn't matter
    # submodular_clf_edge.w += 10*np.ones(submodular_clf_edge.w.shape)
    # print len(submodular_clf_edge.w), submodular_clf_edge.w

    Y_pred = submodular_clf_edge.predict(X_edge)
    assert_array_equal(Y_edge, Y_pred)

    # try to fit the model with non-negativity constraint on the off-diagonal potential, this time
    # with inverted sign on the edge features
    X_edge_neg = [ (x[0], x[1], -x[2]) for x in X_edge ]
    submodular_clf_edge = SubgradientSSVM(model=crf_edge, max_iter=100, C=1,
                                verbose=1,
                                zero_constraint=[4,7],
                                negativity_constraint=[5,6],
                                )
    submodular_clf_edge.fit(X_edge_neg, Y_edge)
    Y_pred = submodular_clf_edge.predict(X_edge_neg)

    assert_array_equal(Y_edge, Y_pred)


def test_binary_ssvm_attractive_potentials_edgefeaturegraph_gc():
    return test_binary_ssvm_attractive_potentials_edgefeaturegraph(inference_method=('ogm', {'alg': 'gc'}))

if __name__=='__main__':
   test_multinomial_blocks_subgradient()
    #test_multinomial_blocks_subgradient()
