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











df=pd.read_csv('new_result_equal.csv',encoding='utf-8-sig')
#df=df.replace('0','')
df=df.fillna('')
#target_test=pd.read_csv('target_test.csv',encoding='utf-8-sig')
#data_test=pd.read_csv('data_test.csv',encoding='utf-8-sig')

#target_test=target_test.fillna('')
#data_test=data_test.fillna('')


#df=df[df['Applied_School']=='剑桥大学']
label=df['Result']
#df['TOEFL']=df['TOEFL'].factorize()[0]
#print (label.value_counts())
#encoder=preprocessing.LabelEncoder()

#labels=encoder.fit_transform(label)

df1=df.drop('Result',axis=1)
df2=df1.drop('Other_Info',axis=1)
df2=df2.drop('Year',axis=1)
df2=df2.drop('TOEFL',axis=1)
df2=df2.drop('IELTS',axis=1)
#df2=df2.drop('School_Major',axis=1)
#df2=df2.drop('English_Level',axis=1)
#df2=df2.drop('Post_School',axis=1)
#df2=df2.drop('Post_Major',axis=1)
#df2=df2.drop('Post_GPA',axis=1)


#df2=df2.drop('Degree',axis=1)



encoder=preprocessing.LabelEncoder()
labels=encoder.fit_transform(label)
labels=pd.DataFrame(labels,columns=['Result'])

#labels=pd.Series(labels)









featurelist=df2 
vec=DictVectorizer()
categorical_fea=['AppliedSchool','Degree','Major','School','SchoolMajor','PostSchool','PostMajor','PostGPA']
#categorical_fea=['AppliedSchool','Degree','Major','School','SchoolMajor']
#vec1=OneHotEncoder(categorical_features=categorical_fea)
#featurelist=vec.fit_transform(featurelist.to_dict(orient='records')).toarray()
#featurelist=vec1.fit_transform(featurelist)
featurelist=pd.get_dummies(featurelist,columns=categorical_fea)
observe=pd.concat([labels,featurelist],axis=1)
observe=observe.drop('Result',axis=1)
observe_name=observe.columns



#data_train=observe
#target_train=labels
#data_train,data_test,target_train,target_test=train_test_split(X_sampled,y_sampled,test_size=0.25)

data_train,data_test,target_train,target_test=train_test_split(observe,labels,test_size=0.25)


##################################SMOTE OVERSAMPLING处理平衡训练集#################################################
#sm=SMOTE(kind='borderline2')
#data_train,target_train=sm.fit_sample(data_train,target_train)
#
#data_train_df=pd.DataFrame(data_train,columns=observe.columns)
#target_train_df=pd.DataFrame(target_train,columns=['Result'])
#
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
#
#
#data_test_df=pd.DataFrame(data_test,columns=labels.columns)
#target_test_df=pd.DataFrame(target_test,columns=labels.columns)


##################################ADASYN OVERSAMPLING处理平衡训练集#################################################
#ada = ADASYN(n_jobs=2)
#data_train,target_train=ada.fit_sample(data_train,target_train)
#
#data_train_df=pd.DataFrame(data_train,columns=observe.columns)
#target_train_df=pd.DataFrame(target_train,columns=['Result'])
#
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
#
#
#data_test_df=pd.DataFrame(data_test,columns=labels.columns)
#target_test_df=pd.DataFrame(target_test,columns=labels.columns)



####################################under sampling ######################################
######ClusterCentroids ######
#cc = ClusterCentroids(n_jobs=1)
#data_train,target_train=cc.fit_sample(data_train,target_train)
#data_train_df=pd.DataFrame(data_train,columns=observe.columns)
#target_train_df=pd.DataFrame(target_train,columns=['Result'])
#
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
#
#data_test_df=pd.DataFrame(data_test,columns=labels.columns)
#target_test_df=pd.DataFrame(target_test,columns=labels.columns)











#featurelist=df2
#enc=pd.get_dummies()
#featurelist=enc
#data_train,data_test,target_train,target_test=train_test_split(featurelist,labels,test_size=0.25)


#print (labels.shape)
#print (featurelist.shape)
estimators = {}
#estimators['bayes'] = GaussianNB()
#estimators['tree'] = tree.DecisionTreeClassifier()

#t1_params = {
#    silent=True, objective='multi:softprob', nthread=-1,
#    max_depth=4, learning_rate= 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6, 'colsample_bylevel': 1,
#    'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0,
#    'reg_alpha': 1, 'reg_lambda': 0,
#    'scale_pos_weight': 3645/25317, 'base_score': 0.5, 'missing': None}


#estimators['forest_100'] = RandomForestClassifier(n_estimators =500,oob_score=True,random_state=5,max_features='auto',min_samples_leaf=5,n_jobs=-1,class_weight='balanced')
estimators['xgboost']= xgb.XGBClassifier(subsample=0.8,colsample_bytree=0.8,learning_rate =0.2,max_depth=30,objective='binary:logistic',nthread=-1,
                                        base_score=0.5,min_child_weight=1,n_estimators=200,reg_lambda=1)

#,scale_pos_weight=2
#estimators['tree']=tree.DecisionTreeClassifier(random_state=2,class_weight='balanced',min_samples_leaf=2,
#                                        max_depth=100,splitter='random',min_samples_split=2)
#cvresult=xgb.cv(estimators['xgboost'].get_xgb_params,)
#estimators['forest_10'] = RandomForestClassifier(n_estimators = 10)
#estimators['svm_c_rbf'] = svm.SVC(class_weight='balanced') 
#estimators['svm_c_linear'] = svm.SVC(kernel='linear',class_weight='balanced')
#estimators['svm_linear'] = svm.LinearSVC()
#estimators['svm_nusvc'] = svm.NuSVC() = [int(i) for i in nums]
start_time=datetime.datetime.now()

#parameters={'n_estimators': [100, 1000], 'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_leaf':[1,20],'random_state':[1,20] }
#gridsearch = GridSearchCV(estimators['forest_100'],param_grid=parameters,cv=10)
#gridsearch.fit(featurelist,labels)
#print (gridsearch.best_params_,gridsearch.best_scores,gridsearch.best_estimator_)



#    
    

for k in estimators.keys():
    start_time = datetime.datetime.now()
    print ('----%s----' % k)
    estimators[k] = estimators[k].fit(data_train, target_train)
    pred = estimators[k].predict(data_test)
    prob=estimators[k].predict_proba(data_test)
    print("%s Score: %0.2f" % (k, estimators[k].score(data_test, target_test)))
#
    xx=target_test
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("%s Time: %0.2f" % (k, time_spend.total_seconds()))
    
    


#    
#    


#    

#    dot_data=tree.export_graphviz(estimators[k],feature_names=vec.get_feature_names())
#    graph=pydotplus.graph_from_dot_data(dot_data)
#    graph.write_pdf('tree.pdf')




    print ('##################NORMAL %s########################'%k)
    print (metrics.classification_report(target_test,pred)) 
#    
    
    
    
    print ('###########%s 权重为#############'%k)
    imp=list(zip(observe.columns, estimators[k].feature_importances_))

#
#    imp=list(zip(vec.feature_names_, estimators[k].feature_importances_))
    impp=pd.DataFrame(imp,columns=['features','importance'])
    f_impp=impp['features']
    line='_'
    for index,value in enumerate(f_impp):
        if line in value:
            f=value.split('_')[0]#+'_'+value.split('_')[1]
            f_impp[index]=f
    impp['features']=f_impp
    Applied_School_imp=impp[impp['features']=='AppliedSchool']
    Degree_imp=impp[impp['features']=='Degree']
    Major_imp=impp[impp['features']=='Major']
    Year_imp=impp[impp['features']=='Year']
    #TOEFL_imp=impp[impp['features']=='TOEFL']
    #IELTS_imp=impp[impp['features']=='IELTS']
    GRE_imp=impp[impp['features']=='GRE']
    School_imp=impp[impp['features']=='School']
    School_Major_imp=impp[impp['features']=='SchoolMajor']
    GPA_imp=impp[impp['features']=='GPA']
    PostSchool_imp=impp[impp['features']=='PostSchool']
    PostMajor_imp=impp[impp['features']=='PostMajor']
    PostGPA_imp=impp[impp['features']=='PostGPA']
    EnglishLevel_imp=impp[impp['features']=='EnglishLevel']
    
    
    
    cal0=Applied_School_imp['importance']
    cal1=Degree_imp['importance']
    cal2=Major_imp['importance']
    #cal3=Year_imp['importance']
    #cal4=TOEFL_imp['importance']
    #cal5=IELTS_imp['importance']
    cal6=GRE_imp['importance']
    cal7=School_imp['importance']
    cal8=School_Major_imp['importance']
    cal9=GPA_imp['importance']
    cal10=PostSchool_imp['importance']
    cal11=PostMajor_imp['importance']
    cal12=PostGPA_imp['importance']
    cal13=EnglishLevel_imp['importance']

    #s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12=0
    
    s0=sum(cal0)
    s1=sum(cal1)
    s2=sum(cal2)
    #s3=sum(cal3)
    #s4=sum(cal4)
    #s5=sum(cal5)
    s6=sum(cal6)
    s7=sum(cal7)
    s8=sum(cal8)
    s9=sum(cal9)
    s10=sum(cal10)
    s11=sum(cal11)
    s12=sum(cal12)
    s13=sum(cal13)
    

        
#    applied_school_cal=s0/cal0.count()
#    degree_cal=s1/cal1.count()
#    major_cal=s2/cal2.count()
#    #year_cal=s3/cal3.count()
#    toefl_cal=s4/cal4.count()
#    ielts_cal=s5/cal5.count()
#    gre_cal=s6/cal6.count()
#    school_cal=s7/cal7.count()
#    school_major_cal=s8/cal8.count()
#    gpa_cal=s9/cal9.count()
##    post_school_cal=s10/cal10.count()
##    post_major_cal=s11/cal11.count()
##    post_gpa_cal=s12/cal12.count()
#    english_level_cal=s13/cal13.count()

    importance_dict=dict((['Applied_School',s0],
                         ['Degree',s1],
                         ['Major',s2],
                         #['Year',year_cal],
                         #['TOEFL',toefl_cal],
                         #['IELTS',ielts_cal],
                         ['GRE',s6],
                         ['School',s7],
                         ['School_Major',s8],
                         ['GPA',s9],
                         ['Post_School',s10],
                         ['Post_Major',s11],
                         ['Post_GPA',s12],
                         ['English_Level',s13]))
##    
#    
#
#    
#    
#    
    ##############################draw plot for metrics###########################
    precision, recall, thresholds=precision_recall_curve(target_test,prob[:,1])
    average_precision=average_precision_score(target_test,prob[:,1])
    fpr,tpr,thresholds=metrics.roc_curve(target_test,prob[:,1])
    roc_auc=auc(fpr,tpr)
    plt.figure(1)
    plt.subplot(221)
    plt.title('ROC CURVE')
    plt.plot(fpr,tpr,'b',label='%s area =%0.2f' % (k,roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.subplot(222)
    plt.title('PR CURVE')
    plt.plot(recall,precision,'r',label='%s area=%0.2f' % (k,average_precision) )
    plt.legend(loc='lower right')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    
    plt.subplot(212)
    plt.bar(range(len(importance_dict)),importance_dict.values(),align='center')
    plt.xticks(range(len(importance_dict)),importance_dict.keys())
    
    plt.show()

#
    print ('###########%s AUC值为#############'%k)
    print (roc_auc)
    print ('###########%s PR AREA值为#############'%k)
    print (average_precision)
    print ('#############%s 混肴矩阵为##############'%k)
    matrix=confusion_matrix(target_test,pred)
    print (matrix)

#    test=pd.DataFrame({'AppliedSchool':'加州大学伯克利分校','Degree':'硕士','Major':'Computer Science','SchoolMajor':'计算机科学','GRE':325,'EnglishLevel':2,'School':'211 & 985','GPA':90},index=[0])
#    test={'Applied_School':'哥伦比亚大学','Degree':'硕士','Major':'Computer Science','TOEFL':90,'School':'211','GPA':80,'GRE':305}
#    test = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
                    #'C': [1, 2, 3]})

#
#test1=pd.get_dummies(test)
#test2=test1.reindex(columns=observe.columns,fill_value=0)
#estimators['xgboost'].predict_proba(test2)







    
    
    
    
    
    
    
    
    
    
