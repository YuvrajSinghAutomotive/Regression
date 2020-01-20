# Histogram: Pure python
# Specify tuple of data
a = (0, 1, 1, 1, 2, 3, 7, 7, 23)

#############################################################################
# Create Histograms Without Bins (Using Counting)
# count instances of observations in a dictionary
def count_elements(seq) -> dict:
     """Tally elements from `seq`."""
     hist = {}
     for i in seq:
         hist[i] = hist.get(i, 0) + 1
     return hist
counted = count_elements(a)
print(counted)

# Use collections.Counter standard library to count elements 
from collections import Counter
recounted = Counter(a)
print(recounted)
print(recounted.items()==counted.items())

# 'plot' ASCII histogram: define a function to create a sorted frequency plot
def ascii_histogram(seq) -> None:
    """A horizontal frequency-table/histogram plot."""
    counted = count_elements(seq)
    for k in sorted(counted):
        print('{0:5d} {1}'.format(k, '+' * counted[k]))

ascii_histogram(a)
print()

#############################################################################
#  plot histogram for pseudorandom numbers with a seed
import random
random.seed(1)
vals = [1, 3, 4, 6, 8, 9, 10]
# Each number in `vals` will occur between 5 and 15 times.
freq = (random.randint(5, 15) for _ in vals)
data=[]
for f,v in zip(freq,vals):
    data.extend([v]*f)
ascii_histogram(data)
print()

#############################################################################

# Creating Bins
# [For understanding] calculating bins from scratch using numpy
import numpy as np
# leftmost and rightmost bin edges
a = np.array(a)
first_edge,last_edge = a.min(),a.max()
n_equal_bins = 10   # numpy's default number of bins
bin_edges = np.linspace(start=first_edge, stop=last_edge, num=n_equal_bins+1, endpoint=True)
print(bin_edges)
bcounts = np.bincount(a)
print(bcounts)

# Using numpy inbuilt histogram() function to calculate bins
hist, _ = np.histogram(a, range=(a.min(),a.max()), bins=a.max()+1)
