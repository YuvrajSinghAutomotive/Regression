# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data (generate random predictor data)
X_raw = np.random.random(100*9)
X_raw = np.reshape(X_raw,(100,9))
# Standard scaler
scaler = StandardScaler().fit(X_raw)
X = scaler.transform(X_raw)
# Add an intercept column at the beginning of predictor matrix
X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
print(np.around(X,2))