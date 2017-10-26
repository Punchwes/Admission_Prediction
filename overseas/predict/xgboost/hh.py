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
import pandas_ml as pdml


new_df=pd.read_csv('new_result_missing.csv',encoding='utf-8-sig')
df_offer=new_df[new_df['Result']=='录取']
df_random=df_offer.sample(frac=0.2)
s=0

while df_random['AppliedSchool'].unique().size<400:
        df_random=df_offer.sample(frac=0.2)
        s=s+1
        print ('次数%d size%s'%(s,df_random['AppliedSchool'].unique().size))
        
print ('done')
#df_random.to_csv('result_offer.csv',header=True,index=False,encoding='utf-8')


#label=new_df['Result']
#
#df1=new_df.drop('Result',axis=1)
#df2=df1.drop('Other_Info',axis=1)
#df2=df2.drop('Year',axis=1)
#df2=df2.drop('TOEFL',axis=1)
#df2=df2.drop('IELTS',axis=1)
##df2=df2.drop('School_Major',axis=1)
##df2=df2.drop('English_Level',axis=1)
#df2=df2.drop('Post_School',axis=1)
#df2=df2.drop('Post_Major',axis=1)
#df2=df2.drop('Post_GPA',axis=1)

#new_df=new_df.fillna('')


#encoder=preprocessing.LabelEncoder()
#labels=encoder.fit_transform(label)
#labels=pd.DataFrame(labels,columns=['Result'])
#
#
#featurelist=df2 
#vec=DictVectorizer()
#categorical_fea=['AppliedSchool','Degree','Major','School','SchoolMajor']
#
#featurelist=pd.get_dummies(featurelist,columns=categorical_fea)
#observe=pd.concat([labels,featurelist],axis=1)
#observe=observe.drop('Result',axis=1)




#sm=SMOTE()
#X_sampled,label_sampled=sm.fit_sample(observe,labels)



#data_train,data_test,target_train,target_test=train_test_split(observe,labels,test_size=0.25)


#encoder=preprocessing.LabelEncoder()
#labels=encoder.fit_transform(label)
#labels=pd.DataFrame(labels,columns=['Result'])
#
#featurelist=df2 
#vec=DictVectorizer()
#categorical_fea=['AppliedSchool','Degree','Major','School','SchoolMajor']
#
#featurelist=pd.get_dummies(featurelist,columns=categorical_fea)
#observe=pd.concat([labels,featurelist],axis=1)
#observe=observe.drop('Result',axis=1)
#observe_name=observe.columns
#
#data_train,data_test,target_train,target_test=train_test_split(observe,labels,test_size=0.08)

#sm=SMOTE()
#data_train,target_train=sm.fit_sample(data_train,target_train)
#
#data_train_df=pd.DataFrame(data_train,columns=observe.columns)
#target_train_df=pd.DataFrame(target_train,columns=['Result'])

#data_eng=data_train_df['EnglishLevel']
#for index,value in enumerate(data_eng):
#    if value>0 and value<0.5:
#        value=0
#    elif value >=0.5 and value <=1:
#        value=1
#    elif value >1 and value <1.5:
#        value=1
#    elif value >=1.5 and value<=2:
#        value=2
#    elif value >2 and value <2.5:
#        value =2
#    elif value >=2.5 and value <=3:
#        value =3
#    elif value >3 and value <3.5:
#        value =3
#    elif value >=3.5 and value<=4:
#        value =4
#    data_eng[index]=value
#data_train_df['EnglishLevel']=data_eng
#        
#data_train_df[(data_train_df>0) & (data_train_df <1)] =1















