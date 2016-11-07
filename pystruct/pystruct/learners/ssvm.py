
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
from pystruct.models.graph_crf import GraphCRF

from ..utils import inference, objective_primal
import copy


class BaseSSVM(BaseEstimator):
    """ABC that implements common functionality."""
    def __init__(self, model, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss_every=0, logger=None):
        self.model = model
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss_every = show_loss_every
        self.n_jobs = n_jobs
        self.logger = logger

    def predict(self, X, scale_features=False):
        """Predict output on examples in X.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.

        Returns
        -------
        Y_pred : list
            List of inference results for X using the learned parameters.

        """
        if scale_features:
            X = BaseSSVM.scale_features(self, copy.deepcopy(X))

        verbose = max(0, self.verbose - 3)
        if self.n_jobs != 1:
            prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(inference)(self.model, x, self.w, idx=idx) for idx, x in enumerate(X))
            return prediction
        else:
            if hasattr(self.model, 'batch_inference'):
                return self.model.batch_inference(X, self.w)
            return [self.model.inference(x, self.w) for x in X]

    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of model.loss)
        over X and Y.

        Parameters
        ----------
        X : iterable
            Evaluation data.

        Y : iterable
            True labels.

        Returns
        -------
        score : float
            Average of 1 - loss over training examples.
        """
        if hasattr(self.model, 'batch_loss'):
            losses = self.model.batch_loss(Y, self.predict(X))
        else:
            losses = [self.model.loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    def _compute_training_loss(self, X, Y, iteration):
        # optionally compute training loss for output / training curve
        if (self.show_loss_every != 0
                and not iteration % self.show_loss_every):
            if not hasattr(self, 'loss_curve_'):
                self.loss_curve_ = []
            display_loss = 1 - self.score(X, Y)
            if self.verbose > 0:
                print("current loss: %f" % (display_loss))
            self.loss_curve_.append(display_loss)

    def _objective(self, X, Y):
        if type(self).__name__ == 'OneSlackSSVM':
            variant = 'one_slack'
        else:
            variant = 'n_slack'
        return objective_primal(self.model, self.w, X, Y, self.C,
                                variant=variant, n_jobs=self.n_jobs)

    @staticmethod
    def get_feature_scaling(ssvm, X):
        if not isinstance(ssvm.model, GraphCRF):
            raise NotImplementedError, "cannot handle models other than GraphCRF"

        all_unaries = np.empty((0,ssvm.model._get_features(X[0]).shape[1]))
        for x in X:
            all_unaries = np.vstack([all_unaries, ssvm.model._get_features(x)])

        unaries_scales = [[], []]
        for i in range(all_unaries.shape[1]):
            unaries_scales[0].append(np.mean(all_unaries[:,i]))
            unaries_scales[1].append(np.std(all_unaries[:,i]))
            if unaries_scales[1][-1] == 0.:
                unaries_scales[1][-1] = 1.

        del all_unaries

        if not isinstance(ssvm.model, EdgeFeatureGraphCRF):
            return unaries_scales, None

        all_pairwise = np.empty((0,ssvm.model._get_edge_features(X[0]).shape[1]))
        for x in X:
            all_pairwise = np.vstack([all_pairwise, ssvm.model._get_edge_features(x)])

        pairwise_scales = [[], []]
        for i in range(all_pairwise.shape[1]):
            pairwise_scales[0].append(np.mean(all_pairwise[:,i]))
            pairwise_scales[1].append(np.std(all_pairwise[:,i]))
            if pairwise_scales[1][-1] == 0.:
                pairwise_scales[1][-1] = 1.

        del all_pairwise

        return unaries_scales, pairwise_scales

    @staticmethod
    def scale_features(ssvm, X):
        if len(ssvm.unaries_scales) == 0 or len(ssvm.pairwise_scales) == 0 \
                or len(ssvm.unaries_scales[0]) == 0 or len(ssvm.pairwise_scales[0]) == 0:
            raise Exception, "the feature scales have not been computed before!"

        if not isinstance(ssvm.model, EdgeFeatureGraphCRF):
            raise NotImplementedError, "cannot handle models other than EdgeFeatureGraphCRF"

        for i, x in enumerate(X):
            X[i] = list(X[i])
            X[i][0] = (x[0]) / ssvm.unaries_scales[1]
            X[i][2] = (x[2]) / ssvm.pairwise_scales[1]

        return X
