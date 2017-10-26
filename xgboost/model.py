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
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn import metrics
import seaborn as sns
import importlib,sys
from scipy import interp
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import ADASYN 
from imblearn.under_sampling import     NearMiss 
import pickle
from sklearn.externals import joblib

df=pd.read_csv('new_result_missing.csv',encoding='utf-8-sig')
df=df.fillna('')

label=df['Result']

df1=df.drop('Result',axis=1)
df2=df1.drop('Other_Info',axis=1)
df2=df2.drop('Year',axis=1)
df2=df2.drop('TOEFL',axis=1)
df2=df2.drop('IELTS',axis=1)
df2=df2.drop('PostSchool',axis=1)
df2=df2.drop('PostMajor',axis=1)
df2=df2.drop('PostGPA',axis=1)

encoder=preprocessing.LabelEncoder()
labels=encoder.fit_transform(label)
labels1=encoder.fit_transform(label)
labels=pd.DataFrame(labels,columns=['Result'])

featurelist=df2 
vec=DictVectorizer()

categorical_fea=['AppliedSchool','Degree','Major','School','SchoolMajor']
featurelist=pd.get_dummies(featurelist,columns=categorical_fea)
featurelist.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)
observe=pd.concat([labels,featurelist],axis=1)

observe_data=observe.drop('Result',axis=1)
observe_name=observe.columns


#observe_decline=observe[observe['Result']==1]
#observe_offer=observe[observe['Result']==0]
#observe_decline_test=observe_decline.sample(frac=0.25)
#observe_offer_test=observe_offer.sample(frac=0.04)
#observe_test=pd.concat([observe_decline_test,observe_offer_test])
#observe_train=pd.concat([observe, observe_test]).drop_duplicates(keep=False)
#
#data_train=observe_train.drop('Result',axis=1)
#data_test=observe_test.drop('Result',axis=1)
#
#target_train=observe_train['Result']
#target_test=observe_test['Result']


###############under sampling#################

new_data_decline=observe[observe['Result']==1] 
new_data_offer=observe[observe['Result']==0]
new_data_offer_random=new_data_offer.sample(frac=0.15)
data=pd.concat([new_data_decline,new_data_offer_random])
data_train=data.drop('Result',axis=1)
target_train=data['Result']

#new_data=pd.concat([data_train,target_train],axis=1)
#new_data_decline=new_data[new_data['Result']==1] 
#new_data_offer=new_data[new_data['Result']==0]
#new_data_offer_random=new_data_offer.sample(frac=0.15)
#data=pd.concat([new_data_decline,new_data_offer_random])
#data_train=data.drop('Result',axis=1)
#target_train=data['Result']



estimator= xgb.XGBClassifier(subsample=0.8,colsample_bytree=0.8,learning_rate =0.1,max_depth=30,objective='binary:logistic',nthread=-1,
                                        base_score=0.5,min_child_weight=1,n_estimators=300)

estimator = estimator.fit(data_train,target_train)

joblib.dump(estimator,'predict.pkl')














