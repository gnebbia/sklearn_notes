#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 gnebbia <nebbionegiuseppe@gmail.com>
#
# Distributed under terms of the GPLv3 license.

"""
We can adjust threshold mainly in three different ways:
    receiver operating characteristics (ROC) 
    Youden's J statistic 
    other methods such as a search with a genetic algorithm.
----------

Comment on AUC:
Concerning the AUC, a simple rule of thumb to evaluate a classifier based on this summary value is the following:
    .90-1 = very good (A)
    .80-.90 = good (B)
    .70-.80 = not so good (C)
    .60-.70 = poor (D)
    .50-.60 = fail (F)
"""

"""
To change threshold we can do:

y_pred = (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool) # set threshold as 0.3
"""

import numpy as np
from sklearn import metrics
from sklearn.datasets import make_classification                                        
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=5)
Xtrain = X[:9000]   
Xtest = X[9000:]    
ytrain = y[:9000]                                        
ytest = y[9000:]      
                                                    
clf = LogisticRegression()               
clf.fit(Xtrain, ytrain)

preds = clf.predict_proba(Xtest)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(ytest, preds)
roc_auc = metrics.auc(fpr,tpr) 


## Optimal Threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
opt_thr_index = np.where(thresholds == optimal_threshold)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

# Insert marker for the optimal threshold
plt.plot(fpr[opt_thr_index],tpr[opt_thr_index],marker='o', color='b')

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




