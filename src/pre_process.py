# Functions to normalize the data 
# Function to shuffle the data
# Function to augment the data
# Function for learning rate finder


import numpy as np

from sklearn.preprocessing import OneHotEncoder

def normalize_data(values):
    # normalize based on zero mean/unit variance
    return (values / 255)
    
def shuffle_data(values, labels):

    values = np.array(values)
    labels = np.array(labels)

    s = np.arange(values.shape[0])
    np.random.shuffle(s)

    values = values[s]
    labels = labels[s]

    return values, labels

def one_hot_encode(labels):
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    label_enocder = OneHotEncoder()
    label_enocder.fit(labels)
    labels_np = label_enocder.transform(labels).toarray()
    return labels_np
