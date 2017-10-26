 #-*- coding=utf-8 -*-
import csv
import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import Imputer


#first step --read & drop_duplicates
#new_df = old_df[((old_df['C1'] > 0) & (old_df['C1'] < 20)) & ((old_df['C2'] > 0) & (old_df['C2'] < 20)) & ((old_df['C3'] > 0) & (old_df['C3'] < 20))]

#new_df=df.duplicated()
#new_df=df.drop_duplicates()
#new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')
new_df=pd.read_csv('new_result_missing.csv',encoding='utf-8-sig')
new_df=new_df.fillna('')


#new_df['TOEFL']=new_df['TOEFL'].fillna('0')
#new_df=new_df.drop('Result',axis=1)
#new_df=new_df.drop('Other_Info',axis=1)
#new_df=new_df.drop('Year',axis=1)
#new_df=new_df.drop('English_Level',axis=1)
#new_df=new_df.drop('Post_School',axis=1)
#new_df=new_df.drop('Post_Major',axis=1)
#new_df=new_df.drop('Post_GPA',axis=1)
#new_df=new_df.drop('GPA',axis=1)
#new_df=new_df.drop('GRE',axis=1)
#new_df=new_df.drop('IELTS',axis=1)
#new_df=new_df.drop('Major',axis=1)
#new_df=new_df.drop('School_Major',axis=1)

#here=new_df.groupby(['Applied_School','Result'])
#missing_value = Imputer(missing_values='NaN', strategy='mean', axis=0)
#def cust_mean(grp):

########################空值处理(TOEFL IELTS)##################################################
new_df['TOEFL']=new_df['TOEFL'].astype(int)
for index in range(len(new_df['TOEFL'])):
    toefl = new_df['TOEFL'][index]
    apschool = new_df['Applied_School'][index]
    school = new_df['School'][index]
    result = new_df['Result'][index]
    if(toefl == 0):
        df_t2 = new_df[new_df['Applied_School'] == apschool]
        df_t3 = df_t2[df_t2['Result'] == result]
        df_t4 = df_t3[df_t3['School'] == school]
        data=df_t4[df_t4['TOEFL']!=0]
        if(len(data['TOEFL']) > 0):
            new_df['TOEFL'][index] = int(data['TOEFL'].mean())
            
for index in range(len(new_df['GPA'])):
    gpa = new_df['GPA'][index]
    apschool = new_df['Applied_School'][index]
    school = new_df['School'][index]
    result = new_df['Result'][index]
    if(gpa == 0):
        df_t2 = new_df[new_df['Applied_School'] == apschool]
        df_t3 = df_t2[df_t2['Result'] == result]
        df_t4 = df_t3[df_t3['School'] == school]
        data=df_t4[df_t4['GPA']!=0]
        if(len(data['GPA']) > 0):
            new_df['GPA'][index] = float(data['GPA'].mean())
            print (float(data['GPA'].mean()))
            
for index in range(len(new_df['TOEFL'])):
    toefl=new_df['TOEFL'][index]
    ielts=new_df['IELTS'][index]
    english=new_df['English_Level'][index]
    if(toefl != '' or ielts!=''):
        if (toefl>0 and toefl<90) or (ielts >0 and ielts <6):
            english='差'
        elif (toefl>=90 and toefl<100) or (ielts>=6.5 and ielts <7):
            english='中'
        elif (toefl>=100 and toefl<110) or (ielts>=7 and ielts <7.5):
            english='良'
        elif (toefl>=110 and toefl<=120) or (ielts>=7.5 and ielts <=9):
            english='优'
        new_df['English_Level'][index]=english
        print (english,toefl,ielts)

        
    
    
    
#for index,value in enumerate(m):
#    if value >=5.5 and value<5.75:
#        value=5.5
#    elif value >=5.75 and value <6:
#        value=6
#    elif value >=6 and value <6.25:
#        value =6
#    elif value >=6.25 and value <6.5:
#        value =6.5
#    elif value >=6.5 and value <6.75:
#        value =6.5
#    elif value >=6.75 and value <7:
#        value =7
#    elif value >=7 and value<7.25:
#        value =7
#    elif value >=7.25 and value <7.5:
#        value =7.5
#    elif value >=7.5 and value < 7.75:
#        value =7.5
#    elif value >=7.75 and value <8:
#        value = 8
#    elif value >=8 and value <8.25:
#        value =8
#    elif value >=8.25 and value <8.5:
#        value =8.5
#    elif value >=8.5 and value <8.75:
#        value =8.5
#    elif value >=8.75 and value <=9:
#        value =9
#    print ('######'+str(index))
#    print (value)
#    m[index]=value
#new_df['IELTS']=m       

        
        
        
        
#here=new_df.groupby(['Applied_School','Result','School'])
#t_here=here['TOEFL']

#for index,value in enumerate(t_here):
#    print (value)
#print (here)






        
        


#df=new_df.duplicated()


#new_df=new_df.drop_duplicates()
#new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')    



#preprocessing
#application result



#new_df["申请结果"]=new_df["申请结果"].replace(('Wailting list','AD无奖','offer','AD小奖'),('被拒','录取','录取','录取'))
#######################学位修改#########################
#for index,degree in enumerate(df):
#    if degree!='博士' and degree!='本科':
#        df[index]='硕士'
#        print (index)
#        
#print ('finished')
#new_df["学位"]=df

################英语和GRE成绩修改############################
###########带有OVERALL修改
tf=new_df['GRE'].fillna('')
x='Overall: '
for index,line in enumerate(tf):
    if x in line:
        score=line[line.find(x)+len(x):].split(',')[0]
        tf[index]=score

new_df['GRE']=tf
print (new_df['GRE'])

14654
new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')

################不带有OVERALL修改
tf=new_df['GRE'].fillna('')
x='V: '
for index,line in enumerate(tf):
    if (x in line) and (line.find('=')<0):
        nums=re.findall(r'\d+',line)
        #print (nums)
        s=0
        sum_list=[]
        nums = [int(i) for i in nums]
        for num in nums:
            if num>=100 and num<=200:
                sum_list.append(num)
                s=s+1
                
        if s==2:
            score=sum(sum_list)
            tf[index]=score
            print (score)
            print ('###########'+str(index))
            
                        
new_df['GRE']=tf
new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')       


################################本科，研究生学校修改############################
##############手动###########

################################本科，研究生专业修改############################

################################本科，研究生成绩修改############################
###############获取成绩###############################
tf=new_df['本科成绩和算法、排名'].fillna('')
for index,value in enumerate(tf):
    if value!='':
        score=re.findall(r'\d+\.?\d*',value)
        scores=[float(i) for i in score]
        score_list=[]
        s=0
        for x in scores:
            score_list.append(x)
            s=s+1
        if s!=0:
            sum_score=score_list[0]
            tf[index]=sum_score
            print ('############'+str(index))
            print (sum_score)
            
            
new_df['本科成绩和算法、排名']=tf  
new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')  
#     
##############################更改绩点->均分##############
tf=new_df['本科成绩和算法、排名'].fillna('')
for index,value in enumerate(tf):
    if value!='' and value!='S' and value!='AA' and value!='A':
        value=float(value)
        
        if value >2.0 and value <=4.0:
            if value <2.3:
                value = 71
                
            elif value >=2.3 and value < 2.7:
                value = 74
                
            elif value >=2.7 and value <3.0:
                value = 77
                
            elif value >=3.0 and value <3.3:
                value = 81
                
            elif value >=3.3 and value <3.7:
                value =84
                
                
            elif value >=3.7 and value <4.0:
                value =88
            
            elif value >=4.0:
                value = 90
                
            print ('############'+str(index))
            print (value)
            tf[index]=value
#        
new_df['本科成绩和算法、排名']=tf 
print (new_df['研究生成绩和算法、排名']) 
new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')  
print (new_df['本科成绩和算法，排名'])


tf=new_df['研究生成绩和算法、排名'].fillna('')
for index,value in enumerate(tf):
    if value!='' and value!='S' and value!='AA' and value!='A':
        value=float(value)
        
        if value >4.0 and value <=5.0:
            if value <4.2:
                value = 91
                
            elif value >=4.2 and value < 4.5:
                value = 92
                
            elif value >=4.5 and value < 4.7:
                value = 93
                
            elif value >=4.7 and value <4.9:
                value = 94
                
            elif value >=4.9:
                value =95
                
                
            print ('############'+str(index))
            print (value)
            tf[index]=value
#
   
new_df['研究生成绩和算法、排名']=tf  
new_df.to_csv("new_result.csv",header=True,index=False,encoding='utf-8')  

        


#print (new_df['TOEFL'].value_counts())

#print (new_df['本科学校档次'].value_counts())
#print (new_df['本科成绩和算法，排名'].value_counts())
#print (new_df['本科成绩和算法，排名'].value_counts().count())








        
        




    
