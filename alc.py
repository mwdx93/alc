import time
import numpy as np
from .ifox import IFOX

class Liver:
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.lobules = 10
        #self.ini = np.random.randn(self.num_input , self.num_input) * np.sqrt(2 / self.num_input)
        self.Cofactor = np.random.randn(self.num_input , self.lobules) * np.sqrt(2 / self.num_input)
        self.Vitamin = np.random.randn(self.num_output, self.num_output) * np.sqrt(2 / self.num_input)
 
    def set_weights(self, solutions):
        # Calculate the sizes of Cofactor, Cofactor2, and Cofactor3
        size_cof = self.Cofactor.size
        self.Cofactor = solutions[:size_cof].reshape(self.Cofactor.shape)
        self.Vitamin = solutions[size_cof:].reshape(self.Vitamin.shape)
 

        
    def get_weights(self):
        # Concatenate all weights into a single vector
        return np.concatenate([self.Cofactor.flatten(), self.Vitamin.flatten()])

    
    def reaction(self, toxins):
        A = np.zeros((toxins.shape[0], self.num_output))
        for i in range(self.num_output):
            A[:, i] = np.sum(toxins * self.Cofactor[:, i], axis=1) + np.mean(self.Cofactor) 
        actived_A = self.relu(A)


        B = np.zeros((toxins.shape[0], self.num_output))
        for i in range(self.num_output):
            B[:, i] = np.sum(actived_A * self.Vitamin[:, i], axis=1) + np.mean(self.Vitamin) 
        excreted_B = self.softmax(B)

        return excreted_B

    


   

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(x): 
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

   
class ALC:
    def __init__(self, detoxification_cycles, detoxification_power):
        self.detoxification_cycles = detoxification_cycles
        self.detoxification_power = detoxification_power
    
    
    def fit(self, X_train, y_train):
        liver = Liver(X_train.shape[-1], y_train.shape[-1])
        opt = IFOX(self.detoxification_cycles, self.detoxification_power, liver.get_weights().size, -1, 1, liver, X_train, y_train)
        opt.optimize()
        liver.set_weights(opt.best_solution)

        return liver
        
            
