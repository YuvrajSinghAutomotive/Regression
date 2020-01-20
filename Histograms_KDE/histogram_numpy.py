# Tutorial URL:
# https://realpython.com/python-histograms/

import numpy as np

# generate random points and values drawn from Laplace distribution
np.random.seed(444) # this ensures that the same random numbers are generated
np.set_printoptions(precision=3)
d = np.random.laplace(loc=15,scale=3,size=500)

# bin calculation using numpy histogram function 
hist,bin_edges = np.histogram(d)
print(hist)         # frequency counts (10 bins by default)
print(bin_edges)    # bin edges 

# Visualizing histogram using Matplotlib
# numpy histogram() is not required
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=d,bins='auto',color='#0504aa', alpha=0.7,rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
plt.show()

# Visualizing histogram using pandas
import pandas as pd
d = pd.Series(d)
d.plot.hist(grid=True,bins=50,rwidth=0.9,color='#607c8e')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
plt.show()

# Generating a Kernel Density Estimate (pandas library)



