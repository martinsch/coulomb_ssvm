import copy
import numpy as np
import random
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from ..utils import inference, objective_primal
from pystruct.learners.ssvm import BaseSSVM


class MBestBaseSSVM(BaseEstimator):
    def __init__(self, model, M=1, gamma=1, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss_every=0, logger=None, sample_assignment_strategy='all'):
        self.model = model
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss_every = show_loss_every
        self.n_jobs = n_jobs
        self.logger = logger
        self.M = M
        self.gamma = gamma
        self.assignment_strategies = ['all', 'best', 'sampled', 'stochastic']
        if sample_assignment_strategy not in self.assignment_strategies:
            raise NotImplementedError("this sample_assignment_strategy is not yet implemented")
        self.sample_assignment_strategy = sample_assignment_strategy


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
        predictions = []
        for m in xrange(self.M):
            if self.n_jobs != 1:
                prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                    delayed(inference)(self.model, x, self.W[m]) for x in X)
                predictions.append( prediction )
            else:
                if hasattr(self.model, 'batch_inference'):
                    predictions.append( self.model.batch_inference(X, self.W[m]) )
                else:
                    predictions.append( [self.model.inference(x, self.W[m]) for x in X] )
        return predictions

    def _compute_relative_scores(self, X, Y):
        """ Computes the relative loss of learner i to all other learners [M]\i

        :param X: Evaluation data
        :param Y: True labels
        :return: Matrix of relative losses for each training sample, columns normalize to one
        """

        # Compute absolute hammming loss
        losses = np.array(self._get_model_losses(X, Y))
        # normalize the losses
        losses = normalize( losses, axis=0, norm='l1' )
        # compute the scores
        losses = 1 - losses
        # normalize again
        losses = normalize( losses, axis=0, norm='l1' )

        return losses

    def _sample_from_probabilities(self, probs):
        """
        Returns an index of probs randomly sampled from the probabilities in probs

        :param probs: normalized probability vector
        :return:
        """
        rand = np.random.uniform(0, 1)
        cum_probs = 0
        for i in xrange(len(probs)):
            cum_probs += probs[i]
            if rand <= cum_probs:
                return i

        assert np.allclose(cum_probs, 1.)
        return len(probs) - 1


    def _get_clustering_assignment(self, X, n_clusters):
        print 'running KMeans clustering for initialization'
        mean_unary_feats = [np.mean(np.array(X[i][0],dtype=np.float),axis=0).squeeze() for i in xrange(len(X))]
        kmeans = KMeans(n_clusters=n_clusters,random_state=42)
        cluster_assignments = kmeans.fit_predict(mean_unary_feats)
        print cluster_assignments
        assignments = np.zeros((self.M, len(X)))
        for i, ca in enumerate(cluster_assignments):
            assignments[ca, i] = 1
        return assignments


    def _get_sample_assignment(self, X, Y, clustering=False):
        """
        Returns a matrix of sample assignments

        :param X: Evaluation data
        :param Y: True labels
        :return:
        """
        shape = (self.M, len(X))

        if self.sample_assignment_strategy == 'all':
            return np.ones(shape)

        assignments = np.zeros((self.M, len(X)))
        if self.sample_assignment_strategy == 'stochastic':
            for m in xrange(self.M):
                sample_idx = np.random.randint(assignments.shape[1])
                assignments[m, sample_idx] = 1
            return assignments

        if self.sample_assignment_strategy == 'best':
            if clustering:
                assignments = self._get_clustering_assignment(X, self.M)
            else:
                absolute_losses = np.array(self._get_model_losses(X, Y))

                # sample when tie (rather than preferring the first occurrence)
                mins = absolute_losses.min(axis=0)
                argmins = []
                for i in range(assignments.shape[1]):
                    argmins.append(random.choice(np.where(absolute_losses[:,i] == mins[i])[0]))

                for i in xrange(assignments.shape[1]):
                    assignments[argmins[i]][i] = 1
            return assignments

        if self.sample_assignment_strategy == 'sampled':
            if clustering:
                assignments = self._get_clustering_assignment(X, self.M)
            else:
                rel_scores = self._compute_relative_scores(X, Y)
                for i in xrange(assignments.shape[1]):
                    sample_idx = self._sample_from_probabilities(rel_scores[:, i])                
                    assignments[sample_idx][i] = 1
            return assignments

        raise NotImplementedError


    def _get_model_losses(self, X, Y):
        Y_pred = self.predict(X, scale_features=False)
        losses = []
        for m in xrange(self.M):
            if hasattr(self.model, 'batch_loss'):
                losses.append(self.model.batch_loss(Y, Y_pred[m]))
            else:
                losses.append([self.model.loss(y, y_pred)
                          for y, y_pred in zip(Y, Y_pred[m])])
        return losses

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
        losses = self._get_model_losses(X, Y)
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    def _compute_training_loss(self, X, Y, iteration, block_coordinate):
        # optionally compute training loss for output / training curve
        if (self.show_loss_every != 0
                and not iteration % self.show_loss_every):
            if not hasattr(self, 'loss_curve_'):
                self.loss_curve_ = [ [] for i in xrange(self.M) ]
            display_loss = 1 - self.score(X, Y)
            if self.verbose > 0:
                print("current loss: %f" % (display_loss))
            self.loss_curve_[block_coordinate].append(display_loss)

    def _objective(self, X, Y, assignments=None):
        obj = 0
        for m in range(len(self.W)):
            if assignments is not None:
                X_batch = X[assignments[m, :].astype(np.bool)]
                Y_batch = Y[assignments[m, :].astype(np.bool)]
            else:
                X_batch = X
                Y_batch = Y
            if len(X_batch) == 0:
                assert len(Y_batch) == 0
                continue
            obj += objective_primal(self.model, self.W[m], X_batch, Y_batch, self.C,
                                variant='n_slack', n_jobs=self.n_jobs)
        return obj / float(len(self.W))

