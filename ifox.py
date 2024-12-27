import warnings
import numpy as np
from utils.helper_functions import cross_entropy_loss
warnings.filterwarnings('ignore')

class IFOX:
    def __init__(self, epochs, pop_size, sol_dim, lb, ub,liver, X_train,y_train):
        self.epochs = epochs
        self.pop_size = pop_size
        self.sol_dim = sol_dim
        self.lb = lb
        self.ub = ub
        self.solutions = [np.random.uniform(low = lb, high = ub, size = (sol_dim,)) for _ in range(self.pop_size)]
        self.liver = liver
        self.best_score = np.inf
        self.best_solution = self.liver.get_weights()
        self.best_solutions = []

        self.X_train = X_train
        self.y_train = y_train

    
    
    # Objective to be optimized
    def obj(self, sol):
        self.liver.set_weights(sol)
        pred = self.liver.reaction(self.X_train)
        loss = cross_entropy_loss(self.y_train,pred)
        return loss
        
    def optimize(self):
        for epoch in range(1, self.epochs+1):
            # 1 Get best agent for optimized agents
            for sol in self.solutions:
                fitness = self.obj(sol)
                if fitness < self.best_score:
                    self.best_solution = sol
                    self.best_score = fitness

            # Keep track for later evaulations
            self.best_solutions.append(self.best_solution)
            
            # 2- Optimize agents
            min_alpha = 1/(self.epochs*2)
            alpha = min_alpha + (1 - min_alpha) * (1 - epoch / (self.epochs))
            
            t = np.mean(np.random.uniform(0, 1, self.best_solution.size))/2
            dis = 0.5 * self.best_solution 
            jump = 0.5 * 9.81 * t ** 2
            
            # Update population
            for i in range(self.pop_size):
                beta = np.random.uniform(-alpha, alpha, size = self.best_solution.size)
                
                if np.random.rand() < alpha:
                    self.solutions[i] = self.best_solution +  beta * alpha
                else:
                    self.solutions[i] = dis * (beta * alpha) / jump