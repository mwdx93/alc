# -----------------------------------------------------------------------------
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to: Share — copy and redistribute the material in any medium or format
#                  Adapt — remix, transform, and build upon the material for any purpose
#                  The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
# Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
# You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#
# Created by Mahmood A. Jumaah and Tarik A. Rashid 2025
#
# Cite as:
#
# Jumaah MA, Ali YH, Rashid TA. Artificial liver classifier: A new alternative to conventional machine learning models.
# Frontiers in Artificial Intelligence. 2025 Aug 8;8:1639720. DOI: https://doi.org/10.48550/arXiv.2501.08074
#
# Jumaah MA, Ali YH, Rashid TA. 2024. Q-FOX Learning:
# QF-tuner: Breaking Tradition in Reinforcement Learning. DOI: https://doi.org/10.48550/arXiv.2402.16562
#
# Jumaah MA, Ali YH, Rashid TA, Vimal S. FOXANN: A Method for Boosting Neural Network Performance.
# Journal of Soft Computing and Computer Applications. 2024;1(1):2.DOI: https://doi.org/10.48550/arXiv.2407.03369
#
# Jumaah MA, Ali YH, Rashid TA (2025) An improved FOX optimization algorithm using adaptive exploration and exploitation
# for global optimization. PLOS ONE 20(9): e0331965. DOI: https://doi.org/10.1371/journal.pone.0331965

# Jumaah MA, Ali YH, Rashid TA. Efficient Q-learning hyperparameter tuning using FOX optimization algorithm.
# Results in Engineering. 2025 Mar 1;25:104341. DOI: https://doi.org/10.1016/j.rineng.2025.104341
# -----------------------------------------------------------------------------

import warnings
import numpy as np

warnings.filterwarnings("ignore")


class IFOX:
    def __init__(self, epochs, pop_size, sol_dim, lb, ub, liver, X_train, y_train):
        self.epochs = epochs
        self.pop_size = pop_size
        self.sol_dim = sol_dim
        self.lb = lb
        self.ub = ub
        self.solutions = [
            np.random.uniform(low=lb, high=ub, size=(sol_dim,))
            for _ in range(self.pop_size)
        ]
        self.liver = liver
        self.best_score = np.inf
        self.best_solution = self.liver.get_weights()
        self.best_solutions = []

        self.X_train = X_train
        self.y_train = y_train

    # Loss function
    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        logp = -np.log(y_pred[np.arange(n_samples), y_true.argmax(axis=1)])
        loss = np.sum(logp) / n_samples
        return loss

    # Objective to be optimized
    def obj(self, sol):
        self.liver.set_weights(sol)
        pred = self.liver.reaction(self.X_train)
        loss = self.cross_entropy_loss(self.y_train, pred)
        return loss

    def optimize(self):
        for epoch in range(1, self.epochs + 1):
            # 1 Get best agent for optimized agents
            for sol in self.solutions:
                fitness = self.obj(sol)
                if fitness < self.best_score:
                    self.best_solution = sol
                    self.best_score = fitness

            # Keep track for later evaulations
            self.best_solutions.append(self.best_solution)

            # 2- Optimize agents
            min_alpha = 1 / (self.epochs * 2)
            alpha = min_alpha + (1 - min_alpha) * (1 - epoch / (self.epochs))

            t = np.mean(np.random.uniform(0, 1, self.best_solution.size)) / 2
            dis = 0.5 * self.best_solution
            jump = 0.5 * 9.81 * t**2

            # Update population
            for i in range(self.pop_size):
                beta = np.random.uniform(-alpha, alpha, size=self.best_solution.size)

                if np.random.rand() < alpha:
                    self.solutions[i] = self.best_solution + beta * alpha
                else:
                    self.solutions[i] = dis * (beta * alpha) / jump
