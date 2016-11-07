import sys
from time import time
import numpy as np
import multiprocessing
import itertools
import copy

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_even_slices, shuffle
from sklearn.metrics.pairwise import cosine_similarity

from .mbest_ssvm import MBestBaseSSVM
from pystruct.learners.ssvm import BaseSSVM
from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
from pystruct.models.graph_crf import GraphCRF
from ..utils import find_constraint, SaveLogger, eval_func_tuple, find_most_violated

class SubgradientMBestSSVM(MBestBaseSSVM):
    """Based on SubgradientSSVM class, see its implementation for details"""
    def __init__(self, model, M=1, gamma=1.0, max_iter=100, C=1.0, verbose=0, momentum=0.0,
                 learning_rate='auto', n_jobs=1, initialize_w_by='',
                 show_loss_every=0, decay_exponent=1, sample_assignment_strategy='all',
                 break_on_no_constraints=True, logger=None, batch_size=None,
                 decay_t0=10, averaging=None, shuffle=False, online=False,
                 break_on_no_loss_improvement=2, negativity_constraint=None,
                 parallel_inference=True, zero_constraint=None,
                 force_moment=1 # s of the Riesz s-energy; 1 for Coulomb force
                 ):
        MBestBaseSSVM.__init__(self, model, M=M, max_iter=max_iter, C=C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          sample_assignment_strategy=sample_assignment_strategy,
                          logger=logger)

        self.averaging = averaging
        self.break_on_no_constraints = break_on_no_constraints
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.t = 0
        self.decay_exponent = decay_exponent
        self.decay_t0 = decay_t0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.M = M
        self.gamma = gamma
        self.initialize_w_by = initialize_w_by
        if online is False:
            print 'Offline learning, setting batch_size to -1'
            self.batch_size = -1
        self.break_on_no_loss_improvement = break_on_no_loss_improvement
        self.negativity_constraint = negativity_constraint
        self.parallel_inference = parallel_inference
        self.zero_constraint = zero_constraint
        if self.negativity_constraint:
            self.neg_factor = [0,] * len(negativity_constraint)
        else:
            self.neg_factor = None
        self.unaries_scales = [[], []]
        self.pairwise_scales = [[], []]
        self.force_moment = force_moment

    def _get_risk_subgradient(self, djoint_feature, n_samples, w):
        ''' returns the risk force (i.e. subgradient on the regularized risk terms)'''
        return djoint_feature * (self.C / n_samples) + w 

    def _compute_coulomb_potential(self, W):
        E = 0
        for m in xrange(self.M):
            for n in xrange(m+1, self.M):
                nrm = np.linalg.norm(self._normalize(W[m]) - self._normalize(W[n]))
                if nrm < 1e-3:
                    nrm = 1e-3
                E += 1./nrm
        return E * 2

    def _get_scaled_projected_coulomb_forces(self, W, tol=1e-6, effective_lr=1.):
        ''' returns the coulomb forces on every point,
            projected to the unit sphere
            and scaled up to the magnitude of the particles (since they
            actually do not lie on the unit sphere)
        '''
        M = len(W)

        # project w's to unit sphere
        W_bar = []
        for i in range(M):
            W_bar.append(self._normalize(W[i]))

        # compute the Coulomb forces on the unit sphere
        forces = [ np.zeros(W[0].shape) for i in range(M) ]
        for i in range(M):
            for j in range(i+1, M):
                force = self._get_pairwise_forces(W_bar[i], W_bar[j])
                forces[i] += force
                forces[j] -= force

        if M == 2:
            assert np.allclose(forces[0], -forces[1])

        for i in range(M):
            # compute the update step on the unit sphere
            w_new = self._normalize(W_bar[i] + self.gamma * effective_lr * forces[i])
            if self.gamma == 0.:
                assert np.allclose(w_new, W_bar[i])
            f = w_new - W_bar[i]

            if np.linalg.norm(f) <= tol:
                # set f to zero'
                forces[i] = np.zeros(W_bar[i].shape)
            else:
                # scale Coulomb force by the magnitudes of W
                f *= np.linalg.norm(W[i])
                forces[i] = f 


        return forces


    @staticmethod
    def _normalize(x):
        if np.allclose(x, 0.):
            return 0.
        norm = np.linalg.norm(x)
        return x / float(norm)


    def _get_pairwise_forces(self, x1, x2):
        ''' Returns the pairwise Coulomb force F exerted on x1 by x2, both
            assumed to be on the unit sphere.
            The Coulomb force exerted on x2 is -F.
        '''
        diff = x1 - x2
        dist = np.linalg.norm(diff)


        if dist < 1e-4:
            diff = np.ones(diff.shape) * 1e-4
            dist = np.linalg.norm(diff)

        # force is the negative gradient
        return self.force_moment * self._normalize(diff)/pow(dist, self.force_moment + 1)

    @staticmethod
    def _get_normal_component(x1, x2):
        ''' returns the component of x1 normal to x2, where x2 is assumed to be normalized '''
        if not np.allclose(np.linalg.norm(x2), 1.0):
            print 'WARNING: x2 is not normalized: np.linalg.norm(x2) =', np.linalg.norm(x2)
        return x1 - np.dot(x1, x2) * x2

    def _update_coulomb_force(self, W):
        if self.batch_size != -1 and self.gamma != 0.:
            raise NotImplementedError, 'online learning not implemented for diverse models'
        
        if self.decay_exponent == 0:
            effective_lr = self.learning_rate_coulomb_
        else:
            effective_lr = (self.learning_rate_coulomb_
                            / (self.t + self.decay_t0)
                            ** self.decay_exponent)

        coulomb_forces = self._get_scaled_projected_coulomb_forces(W, effective_lr=effective_lr)
        if self.gamma == 0.:
            assert np.allclose(coulomb_forces, 0.)

        for m in range(self.M):
            # coulomb_forces[m] is the scaled up force of P( w + self.gamma * F_i) where
            # P is the projection to the unit sphere
            force = coulomb_forces[m]

            # do not overwrite self.grad_old since it is keeping the gradient of
            # the regularized risk only (without Coulomb force)

            # do not adjust the force by a momentum term (as is done in gradient computation)
            if self.verbose and m == 0:
                print 'effective_lr =', effective_lr, ' |effective force| =', np.linalg.norm(force)
            self.W[m] += force

        self.apply_constraints()


    def _solve_subgradient(self, Djoint_feature, N_samples, W):
        """Do a single subgradient step."""
        for m in range(self.M):
            if Djoint_feature[m] is None or N_samples[m] == 0:
                continue
            grad = self._get_risk_subgradient(Djoint_feature[m], N_samples[m], W[m])

            # gradient is updated with momentum only
            assert self.momentum == 0, "if momentum != 0, we have to think about applying it to Coulomb forces as well"

            self.grad_old[m] = grad

            if self.decay_exponent == 0:
                effective_lr = self.learning_rate_
            else:
                effective_lr = (self.learning_rate_
                                / (self.t + self.decay_t0)
                                ** self.decay_exponent)
            if self.verbose and m == 0:
                print 'effective_lr =', effective_lr, ' |effective grad| =', np.linalg.norm(effective_lr * self.grad_old[m])
            W[m] += effective_lr * self.grad_old[m]

        if self.averaging == 'linear':
            raise NotImplementedError
        elif self.averaging == 'squared':
            raise NotImplementedError
        else:
            self.W = W

        self.apply_constraints()
        return self.W

    def apply_constraints(self, m=None):
        if m is None:
            for m in range(self.M):
                self._apply_constraints(m)
        else:
            self._apply_constraints(m)

    def _apply_constraints(self, m):
        if self.negativity_constraint:
            for i, idx in enumerate(self.negativity_constraint):
                if self.neg_factor[i] * self.W[m][idx] > 0:
                    self.W[m][idx] = 0

        if self.zero_constraint:
            for idx in self.zero_constraint:
                self.W[m][idx] = 0.

    def _check_negativity_features(self, X):
        if self.negativity_constraint:
            # check the non-negativity of features with non-negativity constraint
            assert self.model.n_states == 2, "not yet implemented for n_states > 2"
            if isinstance(self.model, GraphCRF):
                for x in X:
                    unary_feats = self.model._get_features(x)
                    edge_feats = np.empty((0,0))
                    if isinstance(self.model, EdgeFeatureGraphCRF):
                        edge_feats = self.model._get_edge_features(x)

                    for i, w_idx in enumerate(self.negativity_constraint):
                        if w_idx < self.model.n_states * unary_feats.shape[1]:
                            f_idx = unary_feats.shape[1] / self.model.n_states
                            feats = unary_feats[:,f_idx]
                        else:
                            f_idx = (w_idx - unary_feats.shape[1] * self.model.n_states) / (self.model.n_states**2)
                            feats = edge_feats[:,f_idx]
                        if np.all(feats >= 0):
                            if self.neg_factor[i] == 0:
                                self.neg_factor[i] = 1
                            elif self.neg_factor[i] != 1:
                                raise AssertionError, "this feature has changing sign over the datasets"
                        elif np.all(feats <= 0):
                            if self.neg_factor[i] == 0:
                                self.neg_factor[i] = -1
                            elif self.neg_factor[i] != -1:
                                raise AssertionError, "this feature has changing sign over the datasets"
                        else:
                            raise AssertionError, "you cannot use a non-negativity constraint on features with changing sign"
            else:
                raise Exception, "cannot check for non-negativity for models other than GraphCRF"

    def fit(self, X, Y, warm_start=False, initialize=True, fold=0):
        """Learn parameters using subgradient descent.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        constraints : None
            Discarded. Only for API compatibility currently.

        warm_start : boolean, default=False
            Whether to restart a previous fit.

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        print("Training mbest primal subgradient structural SVM")
        print(self)

        try:
            size_feats = self.model.size_joint_feature
        except:
            size_feats = self.model.n_states * self.model.n_features +  self.model.n_states ** 2
        self.W = getattr(self, "W", [ np.random.random(self.model.size_joint_feature) for x in range(self.M) ] )
        self.unaries_scales, self.pairwise_scales = BaseSSVM.get_feature_scaling(self, X)

        X = BaseSSVM.scale_features(self, copy.deepcopy(X))

        if initialize:
            self.model.initialize(X, Y)

        self._check_negativity_features(X)

        first_edge_idx = self.model.n_features * self.model.n_states

        self.grad_old = [ np.zeros(self.model.size_joint_feature) for _ in range(self.M) ]

        self.apply_constraints()

        W = []
        for w in self.W:
            W.append(w.copy()) 

        if not warm_start:
            self.objective_curve_ = []
            self.timestamps_ = [time()]
            if self.learning_rate == "auto":
                self.learning_rate_ = self.C / 100.
                self.learning_rate_coulomb_ = self.learning_rate_
            else:
                self.learning_rate_ = self.learning_rate
                self.learning_rate_coulomb_ = self.learning_rate
        else:
            self.timestamps_ = (np.array(self.timestamps_) - time()).tolist()

        best_result = [None, None, None]
        previous_w = [(None,None),] * min(self.max_iter, 20)
        try:
            iteration = 0
            losses_history = []
            coulomb_potential_old = None
            # catch ctrl+c to stop training
            for iteration in xrange(self.max_iter):
                self.t += 1
                if self.shuffle:
                    X, Y = shuffle(X, Y)
                if self.n_jobs == 1:                    
                    objective, positive_slacks, W, losses, assignments = self._sequential_learning(X, Y, W, iteration=iteration)
                else:
                    raise NotImplementedError
                losses_history.append(losses)
                coulomb_potential = self._compute_coulomb_potential(self.W) * self.gamma
                regularizer = np.sum([ np.sum(self.W[block_coordinate] ** 2) / 2. for block_coordinate in range(self.M) ]) 
                objective = 1./ self.M * objective * self.C + coulomb_potential + 1./ self.M * regularizer

                previous_w[iteration%len(previous_w)] = (copy.deepcopy(self.W), assignments.copy())

                if objective >= sys.maxint:
                    raise Exception, "The objective is way too big, choose smaller parameters!"

                # keep track of best result (subgradient method is not a gradient DESCENT method):
                if best_result[0] is None or best_result[0] >= objective:
                    best_result[0] = objective
                    best_result[1] = copy.deepcopy(self.W)
                    best_result[2] = assignments

                if positive_slacks == 0:
                    print("No additional constraints; iteration %d" % iteration)
                    if self.break_on_no_constraints:
                        break

                if self.break_on_no_loss_improvement > 0:
                    loss_change = False
                    if len(losses_history) >= self.break_on_no_loss_improvement:
                        for m in xrange(self.M):
                            l = None
                            for i in xrange(self.break_on_no_loss_improvement):
                                if l is not None:
                                    if l != losses_history[len(losses_history)-i-1][m]:
                                        loss_change = True
                                        break
                                l = losses_history[len(losses_history)-i-1][m]

                        if loss_change is False and abs(coulomb_potential_old-coulomb_potential) < 0.001 * self.gamma:
                            print("The losses did not change compared to the 2 previous iterations and the "
                                  "Coulomb potential changed by less than 0.001. Break after iteration %d" % iteration)
                            break

                if self.verbose > 0:
                    print("iteration %d" % iteration)
                    print("positive slacks: %d, "
                          "objective: %f, of which "
                          "regularizer: %f "
                          "coulomb_potential: %f " %
                          (positive_slacks, objective, regularizer, coulomb_potential))
                self.timestamps_.append(time() - self.timestamps_[0])
                self.objective_curve_.append(objective)

                if self.verbose > 2:
                    print(self.W)

                for m in range(self.M):
                    self._compute_training_loss(X, Y, iteration, m)
                if self.logger is not None:
                    self.logger(self, iteration)

                coulomb_potential_old = coulomb_potential

        except KeyboardInterrupt:
            pass

        # find best of the previous ten w's (this should actually be done in EVERY iteration, but inference is expensive)
        best = [None, None, None] # [objective, W, assignments]
        for W, assignments in previous_w + [tuple(best_result[1:]),]:
            if W is None:
                continue
            self.W = copy.deepcopy(W)
            coulomb_potential = self._compute_coulomb_potential(self.W) * self.gamma
            obj = self._objective(X,Y,assignments=assignments) + coulomb_potential
            if best[0] is None or obj < best[0]:
                best = [obj, copy.deepcopy(W), assignments]

        self.W = best[1]
        self.assignments = best[2]

        if self.verbose:
            print("Computing final objective")
            coulomb_potential = self._compute_coulomb_potential(self.W) * self.gamma
            regularizer = np.sum([ np.sum(W[block_coordinate] ** 2) / 2. for block_coordinate in range(self.M) ]) 
            objective = self._objective(X, Y, assignments=self.assignments) + coulomb_potential
            print("positive slacks: %d, "
                  "objective: %f, of which "
                  "regularizer: %f "
                  "coulomb_potential: %f " %
                  (positive_slacks, objective, regularizer, coulomb_potential))

            print 'break after iteration', iteration
            for m in xrange(self.M):
                print 'W[' + str(m) + '] = ' + str(self.W[m])


        self.timestamps_.append(time() - self.timestamps_[0])
        if self.logger is not None:
            self.logger(self, 'final')
        if self.verbose:
            if self.verbose and self.n_jobs == 1:
                print("calls to inference: %d" % self.model.inference_calls)

        return self

    def _initialize_w_nslack(self, fn):
        try:
            logger = SaveLogger(fn)
            ssvm = logger.load()
            print 'Successfully loaded pretrained nslack model:', fn
        except:
            print 'Could not find the pretrained model:', fn
            raise Exception, 'the M=1 model should be trained outside, with its best C parameter'

        w = ssvm.w

        W = []
        first_edge_idx = self.model.n_states * self.model.n_features
        W.append(w)
        for m in range(1, self.M):
            # perturb the optimal solution by a uniform distribution around from w-0.5 to w+0.5
            W.append(  w + 0.01 * ( np.random.random(w.shape) - 0.05 ) )
        
        self.W = W
        self.apply_constraints()
        self.unaries_scales = ssvm.unaries_scales
        self.pairwise_scales = ssvm.pairwise_scales

        return self.W

    def _parallel_learning(self, X, Y, W):
        raise NotImplementedError


    def _sequential_learning(self, X, Y, W, iteration=None):
        objective, positive_slacks = 0, 0
        losses = np.zeros(self.M)
        slacks = [0,]* self.M
        if self.batch_size in [None, 1]:
            assignments = self._get_sample_assignment(X, Y, clustering=(iteration<=3))
            N_samples = np.sum(assignments, axis=1).tolist()
            assert len(N_samples) == self.M

            self._update_coulomb_force(W)
            for idx, (x, y) in enumerate(zip(X, Y)):
                Delta_joint_feature = []
                for block_coordinate in range(self.M):
                    if assignments[block_coordinate,idx] == 0:
                        Delta_joint_feature.append(None)
                        continue
                    y_hat, delta_joint_feature, slack, loss = \
                        find_constraint(self.model, x, y, W[block_coordinate])
                    slacks[block_coordinate] += slack
                    losses[block_coordinate] += loss
                    objective += slack
                    if slack > 0:
                        positive_slacks += 1
                    Delta_joint_feature.append(delta_joint_feature)
                W = self._solve_subgradient(Delta_joint_feature, N_samples, W)


        else:
            assignments = self._get_sample_assignment(X, Y, clustering=(iteration<=3))
            N_samples = np.sum(assignments, axis=1).tolist()
            assert len(N_samples) == self.M

            # mini batch learning
            if self.batch_size == -1:
                Batch = []
                for m in xrange(self.M):
                    Batch.append(assignments[m, :].astype(np.bool))
                    assert N_samples[m] == np.sum(Batch[-1])
                # slices = [slice(0, len(X))]
                slices = [Batch]
            else:
                raise NotImplementedError

            self._update_coulomb_force(self.W)

            for Batch in slices:
                Delta_joint_feature = [None, ] * self.M
                
                if self.parallel_inference:
                    pool = multiprocessing.Pool(4) #min(self.M, 4))
                    res = pool.map_async(
                                    eval_func_tuple,
                                    itertools.izip(itertools.repeat(find_most_violated),
                                                        range(self.M), 
                                                         itertools.repeat(X), 
                                                         itertools.repeat(Y),
                                                         Batch,
                                                         itertools.repeat(self.model),
                                                         N_samples,
                                                         self.W
                                                         ))
                    vs, ls, ps, ds = zip(*(res.get()))
                   
                    pool.close()
                    pool.join()
                    for worker in pool._pool:
                        assert not worker.is_alive()

                for m in xrange(self.M):
                    if self.parallel_inference:
                        (v, l, p, d) = (vs[m], ls[m], ps[m], ds[m])
                    else:
                        (v, l, p, d) = eval_func_tuple((find_most_violated, m, X, Y, Batch[m], self.model, N_samples[m], self.W[m]))
                    losses[m] += l
                    objective += v
                    positive_slacks += p
                    Delta_joint_feature[m] = d
                W = self._solve_subgradient(Delta_joint_feature, N_samples, W)

        return objective, positive_slacks, W, losses, assignments
