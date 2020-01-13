# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Load data (generate random predictor data)
X_raw = np.random.random(100*9)
X_raw = np.reshape(X_raw,(100,9))
# Standard scaler
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)
# Add an intercept column at the beginning of predictor matrix
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
print(np.around(X,2))
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
beta_init = np.array([1]*X.shape[1])
result = minimize(objective_function,beta_init,args=(X,Y),method='BFGS',options={'maxiter':500})

beta_hat = result.x
print()   # print(beta_hat)

LinRegRes = pd.DataFrame({
    "true_beta": beta,
    "estimated_beta": beta_hat,
    "error": beta - beta_hat
})[["true_beta","estimated_beta","error"]]
print(LinRegRes)
print()
print("Loss Function value = " + str(loss_function(np.matmul(X,beta_hat),Y,None)))

# Incorporating regularization into model fitting
