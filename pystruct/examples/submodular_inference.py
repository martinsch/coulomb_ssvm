import numpy as np
from numpy.ma.testutils import assert_array_equal
from pystruct.learners import NSlackSSVM
from pystruct.learners.subgradient_ssvm import SubgradientSSVM
from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
from pystruct.datasets import generate_blocks
from pystruct.models import GridCRF
import matplotlib.pyplot as plt


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

# test_binary_ssvm_attractive_potentials_edgefeaturegraph()
test_binary_ssvm_attractive_potentials_edgefeaturegraph(('ogm', {'alg': 'gc'}))
