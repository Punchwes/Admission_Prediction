import csv
import numpy as np
import pandas as pd 
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.model_selection import cross_val_predict
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from IPython.display import Image
import os
import pydotplus
from sklearn.learning_curve import learning_curve
import pylab as pl
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import importlib,sys
from scipy import interp
from sklearn.grid_search import GridSearchCV



df=pd.read_csv('new_result.csv',encoding='utf-8-sig',dtype=object)
df=df.replace('0','')
df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')    

#df=df.fillna('')
#label=df['Result']
#
#df1=df.drop('Result',axis=1)
#df2=df1.drop('Other_Info',axis=1)
#
#
#encoder=preprocessing.LabelEncoder()
#labels=encoder.fit_transform(label)
#
#featurelist=df2
#vec=DictVectorizer()
#featurelist=vec.fit_transform(featurelist.to_dict(orient='records')).toarray()
#data_train,data_test,target_train,target_test=train_test_split(featurelist,labels,test_size=0.25)
#
#
#estimators = {}
#estimators['forest_100'] = RandomForestClassifier(n_estimators =100,oob_score=True,random_state=2,max_features='auto',min_samples_leaf=2,n_jobs=-1,class_weight={0:.12,1:.88})
#
#parameters={'max_features': ['auto', 'sqrt', 'log2'], 
#            'min_samples_leaf':[1,10],
#            'random_state':[1,],
#            'class_weight':[{1:m} for m in [0.7,0.9]]}
#gridsearch = GridSearchCV(estimators['forest_100'],param_grid=parameters,cv=10)
#gridsearch.fit(featurelist,labels)
#print (gridsearch.best_params_,gridsearch.best_score,gridsearch.best_estimator_)























