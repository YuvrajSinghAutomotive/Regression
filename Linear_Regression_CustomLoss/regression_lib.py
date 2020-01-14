import numpy as np
from scipy.optimize import minimize

# LOAD DATA HERE

# LOSS FUNCTIONS
class Loss:
    def __init__(self,y_pred,y_true,sample_weights,delta=None):
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)
        assert len(self.y_true) == len(self.y_pred)

        if type(sample_weights) != type(None):
            self.sample_weights = np.array(sample_weights)
            assert len(self.sample_weights) == len(self.y_true)
        else:
            self.sample_weights = np.full(self.y_true.shape,1)
        
        self.delta = np.array(delta)
        
    # Loss function methods
    def SSE(self):
        error = np.sum(np.multiply(np.square(self.y_true - self.y_pred), self.sample_weights))
        return(error)

    def MSE(self):
        error = np.mean(np.multiply(np.square(self.y_true - self.y_pred), self.sample_weights))
        return(error)

    def RMSE(self):
        error = np.sqrt(self.MSE())
        return(error)

    def SAE(self):
        error = np.sum(np.multiply(np.abs(self.y_true - self.y_pred), self.sample_weights))
        return(error)

    def MAE(self):
        error = np.mean(np.multiply(np.abs(self.y_true - self.y_pred), self.sample_weights))
        return(error)

    def huber(self):
        if type(self.delta) != type(None):
            error = self.MAE()      # modify later after gaining more knowledge of Huber loss
            return(error)
        else:
            print("Huber Delta not defined: Using Mean Absolute Error")
            error = self.MAE()
            return(error)

    # IN CASE YOU WANT TO DEFINE A CUSTOM LOSS FUNCTION
    def custom_loss(self):
        pass

# MODELS
class Model:
    def __init__(self,X,Y,Loss):
        pass
    def objective(self):
        pass
    def fit(self):
        pass

# Define inherited model classes
class Linear(Model):
    def predict(self):
        pass
    def regularization(self):
        pass

class Polynomial(Model):
    def predict(self):
        pass
    def regularization(self):
        pass

# IN CASE YOU NEED TO DEFINE A CUSTOM MODEL
class CustomModel(Model):
    def predict(self):
        pass
    def regularization(self):
        pass

#  MODEL VALIDATION
class Validator:
    def __init__(self,X,Y,weights):
        pass

# Define inherited validation methods
class NoCV(Validator):
    def TestTrainSplit(self,split_ratio):
        pass
    def holdout(self,holdout_ratio): # you have the option of whether you want a holdout set or not
        pass

class CV(Validator):
    def LOOCV(self):
        pass
    def Kfold(self):
        pass
    def MonteCarlo(self):
        pass
