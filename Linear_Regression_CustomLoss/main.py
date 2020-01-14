# import libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

os.system('cls')

# Load data (generate random predictor data)
X_raw = np.random.random(100*9)
X_raw = np.reshape(X_raw,(100,9))
# Standard scaler
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)
# Add an intercept column at the beginning of predictor matrix
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
# print(np.around(X,2))
# Define "true" beta regression coefficients
beta = np.array([2,6,7,3,5,7,1,2,2,8])
# y = Xb
Y_true = np.matmul(X,beta)
# observed data with noise
Y = Y_true*np.exp(np.random.normal(loc=0.0,scale=0.2,size=100))

# Define loss function
def mean_absolute_percentage_error(y_pred,y_true,sample_weights):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    if np.any(y_true==0):
        print("Found zeros in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true,idx)
        y_pred = np.delete(y_pred,idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights,idx)

    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true-y_pred)/y_true))*100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights) * np.dot(sample_weights,np.abs((y_true-y_pred)/y_true)))

loss_function = mean_absolute_percentage_error

# Fitting a linear model
def objective_function(beta,X,Y):
    error = loss_function(np.matmul(X,beta),Y,None)
    return(error)

# You must provide a starting point at which to initialize the parameter search space
beta_init = np.array([1] * X.shape[1])
result = minimize(objective_function,beta_init,args=(X,Y),method='BFGS',options={'maxiter':500})

beta_hat = result.x
print()   # print(beta_hat)

LinRegRes = pd.DataFrame({
    "true_beta": beta,
    "estimated_beta": np.around(beta_hat,2),
    "error": np.around(beta - beta_hat,3)
})[["true_beta","estimated_beta","error"]]
print(LinRegRes)
print()
print("Loss Function value = " + str( np.around(loss_function(np.matmul(X,beta_hat),Y,None),3) ))

# Incorporating regularization into model fitting
class LinearModel_regularized:
    """ 
    Linear Model: Y=Xb, fit by minimizing the provided loss function with l2-regularization
    """
    def __init__(self,loss_function=None,X=None,Y=None,sample_weights=None,beta_init=None,regularization=None,reg_type=None):
        self.regularization = regularization
        self.reg_type = None
        self.beta = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        self.X = X
        self.Y = Y

    def predict(self,X):
        prediction = np.matmul(X,self.beta)
        return(prediction)

    def model_error(self):
        error = self.loss_function(self.predict(self.X),self.Y,sample_weights=self.sample_weights)
        return(error)

    def l2_regularized_loss(self,beta):
        self.beta = beta
        return(self.model_error() + sum(self.regularization * np.array(self.beta)**2))

    def l1_regularized_loss(self,beta):
        self.beta = beta
        return(self.model_error() + sum(self.regularization * np.array(abs(self.beta))))

    def fit(self,maxiter=250):
        if type(self.beta_init)==type(None):
            self.beta_init = np.array([1]*self.X.shape[1])
        else:
            pass

        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more iterations.")

        if self.reg_type == 'L1':
            res = minimize(self.l1_regularized_loss,self.beta_init,args=(self.X,self.Y),method='BFGS',options={'maxiter':500})
            self.beta = res.x
            self.beta_init = self.beta
        if self.reg_type == 'L2':
            res = minimize(self.l2_regularized_loss,self.beta_init,args=(self.X,self.Y),method='BFGS',options={'maxiter':500})
            self.beta = res.x
            self.beta_init = self.beta


print("L2 regularized model")
l2_mape_model = LinearModel_regularized(loss_function=mean_absolute_percentage_error,X=X,Y=Y,regularization=0.001,reg_type='L2')
l2_mape_model.fit()
l2_mape_model.beta