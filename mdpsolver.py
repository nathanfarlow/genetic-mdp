# Note: most of this code is copied/adapted from https://pymoo.org/customization/subset.html?highlight=binarycrossover

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize


class MDPProblem(ElementwiseProblem):

    def __init__(self, points: np.array, subset_size: int, distance_func: callable, **kwargs):
        super().__init__(n_var=len(points), n_obj=1,
                         n_ieq_constr=1, **kwargs)
        self.points = points
        self.subset_size = subset_size
        self.distance_func = distance_func

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = -self._calculate_average_distance(x)
        out['G'] = (self.subset_size - np.sum(x)) ** 2

    def _calculate_average_distance(self, x):
        total = 0
        subset = np.where(x == 1)[0]
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                total += self.distance_func(
                    self.points[subset[i]], self.points[subset[j]])
        return total / (len(subset) * (len(subset) - 1) / 2)


class MDPSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.subset_size]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.subset_size - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MDPMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X


def solve_mdp(points: list, subset_size: int, distance_func: callable, n_gen: int = 100, n_pop: int = 100, verbose: bool = False):
    problem = MDPProblem(points, subset_size, distance_func)
    algorithm = GA(
        pop_size=n_pop,
        sampling=MDPSampling(),
        crossover=BinaryCrossover(),
        mutation=MDPMutation(),
        eliminate_duplicates=True
    )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=verbose,
                   save_history=False)
    return res
