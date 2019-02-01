# -*- coding: utf-8 -*-
"""
It converts the Auslan dataset into two CSV files.

1. data.csv: Each line contains a single example.

Each example is a vector with 22*57 value. 
The first 22 values represent the 22 features in the first frame, and so on.

2. labels.csv: Each line contains a single label, stored as a string. 

"""

import os
import pandas as pd
from scipy import signal


# Collecting all the relative file paths in a single list
files = []
for (dirpath, dirnames, filenames) in os.walk('.'):
    if dirpath != '.':  # Not considering files not in a subdirectory
        files += [(dirpath + '/' + f) for f in filenames]

# Preparing the output files
data = open('data.csv', 'a')
labels = open('labels.csv', 'a')
count = 0

# Processing each file
for f in files:
    # Opening the file
    file = pd.read_csv(f, sep='\t')
    for i in range(22):
        # Obtaining the column of the first feature
        col_data = file.iloc[:, i].values
        # Applying the FFT
        col_data = signal.resample(col_data, 57)
        # Converting the row vector into a column vector
        col_data = col_data.reshape((1, 57))
        # Appending it to the data file
        # print(col_data)
        pd.DataFrame(col_data).to_csv(data, header=False, index=False)
          
    # Getting the label
    print(f)
    l = os.path.basename(f).split('-')[0]
    # Appending the label to the labels file
    labels.write(l + '\n')
    count += 1
    print(count)

data.close()
labels.close()
