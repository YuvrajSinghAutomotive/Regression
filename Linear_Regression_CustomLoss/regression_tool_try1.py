import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

class loss:
    def __init__(self, y_true, y_pred, sample_weights, beta, regularization_type=None,regularization_parameter=0.001):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        assert len(self.y_true) == len(self.y_pred)

        if type(sample_weights) != type(None):
            self.sample_weights = sample_weights
            assert len(self.sample_weights) == len(self.y_true)
        else:
            self.sample_weights = np.full(self.y_true.shape,1)

        self.regularization = self.regularization_loss(beta, regularization_type, regularization_parameter)
    
    def sum_squares_error(self):
        error = np.sum(np.multiply(np.square(self.y_true - self.y_pred), self.sample_weights))
        return(error + self.regularization)
        
    class regularization_loss:
        def __init__(self, beta, regularization_type=None,regularization_parameter):
            self.beta = beta
            self.regularization_type = regularization_type
            self.regularization_parameter = regularization_parameter

            if type(regularization_type) != type(None):
                if regularization_type == 'L1':
                    print(self.L1)
                    return([0])

                elif regularization_type == 'L2':
                    print(self.L2)
                    return([0])

                elif regularization_type == 'ElasticNet':
                    print(self.ElasticNet)
                    return([0])
            else:
                print("No regularization")
                return([0])

        def L1(self):
            return("L1 regularization")

        def L2(self):
            return("L2 regularization")

        def ElasticNet(self):
            return("Elastic Net regularization")


class Model:
    def __init__(self, X, Y, beta_init, loss_function, regularization, regularization_parameter, sample_weights):
        self.X = X
        self.Y = Y
        self.beta_init = np.array(beta_init)
        self.loss_function = loss_function
        self.regularization = regularization
        self.regularization_parameter = regularization_parameter
        self.sample_weights = sample_weights

        self.beta = np.full((self.beta_init.shape),[None])

        assert self.X.shape[0] == self.Y.shape[0]
        assert self.sample_weights.shape[0] == self.Y.shape[0]
        assert self.Y.shape[1] == 1

    class Linear:
        def predict(self):
            prediction = np.matmul(self.X,self.beta)
            return(prediction)

        def fit(self):
            pass
    
class CrossValidator:
    pass