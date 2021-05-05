import typing

import numpy as np
import math
import pandas as pd
from bitarray import bitarray
import time
import heapq

from optimization.c_estimation.base_c_estimator import BaseCEstimator


class TIcEEstimator(BaseCEstimator):
    c_estimates_by_iteration: typing.List[typing.List[float]]
    time: float

    # * -d, --delta DELTA
    #                             Delta, default: using formula from paper.
    # * -k, --max-bepp MAX_BEPP
    #                             The max-bepp parameter k, default=5.
    # * -M, --maxSplits MAXSPLITS
    #                             The maximum number of splits in the decision tree,
    #                             default=500.
    # * --useMostPromisingOnly
    #                             Set this option to only use the most promising subset
    #                             (instead of calculating the maximum lower bound)
    # * -m, --minT MINT
    #                             The minimum set size to update the lower bound with,
    #                             default=10.
    # * -i, --nbIts NBITS
    #                             The number of times to repeat the the estimation
    #                             process. Default 2 (first with c_prior=0.5, then with
    #                             c_prior=c_estimate)
    def __init__(self, k: int = 5, delta=None, nbIts: int = 2,
                 maxSplits: int = 500, useMostPromisingOnly=False, minT: int = 10):
        self.k = k
        self.delta = delta
        self.nbIts = nbIts
        self.maxSplits = maxSplits
        self.useMostPromisingOnly = useMostPromisingOnly
        self.minT = minT

    def fit(self, X, s):
        self.P_s_1 = float(np.mean(s == 1))

        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        labels = bitarray(list(s.astype(int)))
        folds = np.random.randint(5, size=len(X))

        ti = time.time()
        c_estimate, c_its_estimates = self.tice(X, labels, folds)
        ti = time.time() - ti

        self.c_estimate = c_estimate
        self.c_estimates_by_iteration = c_its_estimates
        self.time = ti

        return c_estimate

    def pick_delta(self, T):
        return max(0.025, 1 / (1 + 0.004 * T))

    def low_c(self, data, label, delta, minT, c=0.5):
        T = float(data.count())
        if T < minT:
            return 0.0
        L = float((data & label).count())
        c_low = L / T - math.sqrt(c * (1 - c) * (1 - delta) / (delta * T))
        return c_low

    def max_bepp(self, k):
        def fun(counts):
            return max([(0 if T_P[0] == 0 else float(T_P[1]) / (T_P[0] + k)) for T_P in counts])

        return fun

    def generate_folds(self, folds):
        for fold in range(max(folds) + 1):
            tree_train = bitarray(list((folds == fold).astype(int)))
            estimate = ~tree_train
            yield tree_train, estimate

    def tice(self, data, labels, folds):
        c_its_ests = []
        c_estimate = 0.5

        for it in range(self.nbIts):
            c_estimates = []

            global c_cur_best  # global so that it can be used for optimizing queue.
            for tree_train, estimate in self.generate_folds(folds):
                c_cur_best = self.low_c(estimate, labels, 1.0, self.minT, c=c_estimate)
                cur_delta = self.delta if self.delta else self.pick_delta(estimate.count())

                if self.useMostPromisingOnly:

                    c_tree_best = 0.0
                    most_promising = estimate
                    for tree_subset, estimate_subset in self.subsetsThroughDT(data, tree_train, estimate, labels,
                                                                              splitCrit=self.max_bepp(self.k),
                                                                              minExamples=self.minT,
                                                                              maxSplits=self.maxSplits,
                                                                              c_prior=c_estimate,
                                                                              delta=cur_delta):
                        tree_est_here = self.low_c(tree_subset, labels, cur_delta, 1, c=c_estimate)
                        if tree_est_here > c_tree_best:
                            c_tree_best = tree_est_here
                            most_promising = estimate_subset

                    c_estimates.append(max(c_cur_best, self.low_c(most_promising, labels, cur_delta,
                                                                  self.minT, c=c_estimate)))

                else:
                    for tree_subset, estimate_subset in self.subsetsThroughDT(data, tree_train, estimate, labels,
                                                                              splitCrit=self.max_bepp(self.k),
                                                                              minExamples=self.minT,
                                                                              maxSplits=self.maxSplits,
                                                                              c_prior=c_estimate,
                                                                              delta=cur_delta):
                        est_here = self.low_c(estimate_subset, labels, cur_delta, self.minT, c=c_estimate)
                        c_cur_best = max(c_cur_best, est_here)
                    c_estimates.append(c_cur_best)

            c_estimate = sum(c_estimates) / float(len(c_estimates))
            c_its_ests.append(c_estimates)

        return c_estimate, c_its_ests

    def subsetsThroughDT(self, data, tree_train, estimate, labels, splitCrit, minExamples=10,
                         maxSplits=500, c_prior=0.5, delta=0.0):
        # This learns a decision tree and updates the label frequency lower bound for every tried split.
        # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
        # The input data is expected to have only binary or continues variables with values between 0 and 1.
        # To achieve this, the multivalued variables should be binarized and the continuous variables
        # should be normalized.

        # Max: Return all the subsets encountered

        all_data = tree_train | estimate

        borders = [.25, .5, .75]

        def make_subsets(a):
            subsets = []
            options = bitarray(all_data)
            for b in borders:
                X_cond = bitarray(list((data[:, a] < b).astype(int))) & options
                options &= ~X_cond
                subsets.append(X_cond)
            subsets.append(options)
            return subsets

        conditionSets = [make_subsets(a) for a in range(data.shape[1])]

        priorityq = []
        heapq.heappush(priorityq, (
            -self.low_c(tree_train, labels, delta, 0, c=c_prior), -(tree_train & labels).count(), tree_train, estimate,
            set(range(data.shape[1])), 0
        ))
        yield (tree_train, estimate)

        n = 0
        minimumLabeled = 1
        while n < maxSplits and len(priorityq) > 0:
            n += 1
            (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = heapq.heappop(priorityq)
            lab_count = -neg_lab_count

            best_a = -1
            best_score = -1
            best_subsets_train = []
            best_subsets_estimate = []
            best_lab_counts = []
            uselessAs = set()

            for a in available:
                subsets_train = [X_cond & subset_train for X_cond in conditionSets[a]]
                subsets_estimate = [X_cond & subset_train for X_cond in conditionSets[a]]
                estimate_lab_counts = [(subset & labels).count() for subset in subsets_estimate]
                if max(estimate_lab_counts) < minimumLabeled:
                    uselessAs.add(a)
                else:
                    score = splitCrit([(subsub.count(), (subsub & labels).count()) for subsub in subsets_train])
                    if score > best_score:
                        best_score = score
                        best_a = a
                        best_subsets_train = subsets_train
                        best_subsets_estimate = subsets_estimate
                        best_lab_counts = estimate_lab_counts

            fake_split = len([subset for subset in best_subsets_estimate if subset.count() > 0]) == 1

            if best_score > 0 and not fake_split:
                newAvailable = available - { best_a } - uselessAs
                for subsub_train, subsub_estimate in zip(best_subsets_train, best_subsets_estimate):
                    yield (subsub_train, subsub_estimate)
                minimumLabeled = c_prior * (1 - c_prior) * (1 - delta) / (delta * (1 - c_cur_best) ** 2)

                for (subsub_lab_count, subsub_train, subsub_estimate) in zip(best_lab_counts, best_subsets_train,
                                                                             best_subsets_estimate):
                    if subsub_lab_count > minimumLabeled:
                        total = subsub_train.count()
                        if total > minExamples:  # stop criterion: minimum size for splitting
                            train_lab_count = (subsub_train & labels).count()
                            if lab_count != 0 and lab_count != total:  # stop criterion: purity
                                heapq.heappush(priorityq, (
                                    -self.low_c(subsub_train, labels, delta, 0, c=c_prior), -train_lab_count,
                                    subsub_train, subsub_estimate, newAvailable, depth + 1))
