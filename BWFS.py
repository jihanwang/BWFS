from entropy_estimators import *
import numpy as np
import pickle
import os

# X is features and y is labels
def bwfs(X, y, **kwargs):
    n_samples, n_features = X.shape
    is_n_selected_features_specified = False
    F = [] # the feature index list
    MIfy = [] # the weight list
    t1 = np.zeros(n_features) # record the mutual information between features and labels
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            MIfy.append(t1[idx])

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        sumBWFSForXj = 0
        for j in range(n_features):
            if j not in F:
                t = 0
                for i in F:
                    f_select = X[:, i]
                    f = X[:, j]
                    # entropyd is used for obtaining the entropy of vector
                    H = entropyd(y)+entropyd(f_select)
                    conmuikyj = cmidd(f, y, f_select) #conditional mutual information 
                    conmuijyk = cmidd(f_select, y, f)
                    t += (1+2*(conmuijyk - t1[i])/H)*conmuikyj # the objective function
                if t > sumBWFSForXj:
                    sumBWFSForXj = t
                    idx = j
        F.append(idx)
        MIfy.append(sumBWFSForXj)

    return np.array(F), np.array(MIfy)
