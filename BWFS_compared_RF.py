from entropy_estimators import *
import scipy.io
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import cross_val_score

from libsvm.svmutil import *

def bwfs(X, y, **kwargs):
    n_samples, n_features = X.shape
    is_n_selected_features_specified = False
    F = []
    MIfy = []
    t1 = np.zeros(n_features)
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
                    H = entropyd(y)+entropyd(f_select)
                    muijy = t1[i]
                    conmuikyj = cmidd(f, y, f_select)
                    conmuijyk = cmidd(f_select, y, f)
                    t += (1+2*(conmuijyk - t1[i])/H)*conmuikyj
                if t > sumBWFSForXj:
                    sumBWFSForXj = t
                    idx = j
        F.append(idx)
        MIfy.append(sumBWFSForXj)

    return np.array(F), np.array(MIfy)

def testBySvmlib(x_train,y_train,x_test,y_test,feats):
    prob  = svm_problem(np.array(y_train),np.array(x_train[:,feats]))
    param = svm_parameter('-t 0 -c 4 -b 1')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(np.array(y_test), np.array(x_test[:,feats]), model)
    #print(p_label,p_acc,p_val)
    # print(p_acc[0])
    return p_acc[0]

def testByDecisionTree(x_train,y_train,x_test,y_test,feats):
    Dtc = dtc()
    Dtc.fit(np.array(x_train[:,feats]),np.array(y_train))
    pre = Dtc.predict(np.array(x_test[:,feats]))
    p_acc = sum(pre == y_test)/len(y_test)
    return p_acc

def testByGaussianNB(x_train,y_train,x_test,y_test,feats):
    clf = gnb()
    clf = clf.fit(np.array(x_train[:,feats]),np.array(y_train))
    pre = clf.predict(np.array(x_test[:,feats]))
    p_acc = sum(pre == y_test)/len(y_test) 
    return p_acc

def testByKNeighbors(x_train,y_train,x_test,y_test,feats):
    clf = knc(n_neighbors=3)
    clf.fit(np.array(x_train[:,feats]),np.array(y_train))
    pre = clf.predict(np.array(x_test[:,feats]))
    p_acc = (pre == y_test)/len(y_test) 
    return p_acc

#save the result
def saveMyResult(dName,myDic):
    outputFile = open(dName+'.pkl','wb')
    pickle.dump(myDic,outputFile)
    outputFile.close()

if __name__ == '__main__':
    # load data
    os.chdir(r'D:\skfeature\example')
    datasetName = ['sonar','waveform','splice','madelon','isolet']
    resultSVM={}
    resultNBC={}
    resultRFC={}
    resultDTC={}
    resultKNC={}
    resultFeas={}
    num_fea = 10    # number of selected features
    for dataName in datasetName:
        print(dataName+'--------------------------------------------')
        mat = scipy.io.loadmat('../data/' + dataName + '.mat')
        print(dataName,mat['X'].shape)
        x = mat['X']    # data
        x = x.astype(float)
        y = mat['Y']    # label
        y = y[:, 0]
        n_samples, n_features = x.shape    # number of samples and number of features

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        
        #random forest
        forest = RandomForestClassifier(n_estimators=1000, random_state=0)
        forest.fit(x_train, y_train)
        importances = forest.feature_importances_
        print(importances)
        indices = np.argsort(importances)[::-1] # 下标排序
        print(indices[0:num_fea])
        
        idxDicF = {} # record the selected features by every method
        idxDicF['rf'] = indices[0:num_fea]
        idxDicAccSVM = {}
        idxDicAccNBC = {}
        idxDicAccDTC = {}
        print('start')
        idx,_ = bwfs(x_train, y_train, n_selected_features=num_fea)
        print(idx)
        idxDicF['bwfs'] = idx
        for method, feas in idxDicF.items():
            for i in range(len(feas)):
                fea = feas[0:i+1]
                accDTC = testByDecisionTree(x_train, y_train, x_test,y_test,fea)
                tempAccDTC = idxDicAccDTC.get(method,[]) 
                tempAccDTC.append(accDTC)
                idxDicAccDTC[method] = tempAccDTC

                accNBC = testByGaussianNB(x_train, y_train, x_test,y_test,fea)
                tempAccNBC = idxDicAccNBC.get(method,[])
                tempAccNBC.append(accNBC)
                idxDicAccNBC[method] = tempAccNBC

                accSVM = testBySvmlib(x_train, y_train, x_test,y_test,fea)
                tempAccSVM = idxDicAccSVM.get(method,[]) 
                tempAccSVM.append(accSVM)
                idxDicAccSVM[method] = tempAccSVM
        print(idxDicF)
        resultFeas[dataName] = idxDicF
        resultSVM[dataName] = idxDicAccSVM
        resultNBC[dataName] = idxDicAccNBC
        resultDTC[dataName] = idxDicAccDTC
    saveMyResult('feas',resultFeas)
    saveMyResult('svm',resultSVM)
    saveMyResult('nbc',resultNBC)
    saveMyResult('dtc',resultDTC)
