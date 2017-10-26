from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render,render_to_response
from predict.models import NewResultEqual,NewResultMissing
import json
import pickle
import pandas as pd 
import csv
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb
from sklearn import preprocessing
from django.db.models import Count

def home(request):
	return render(request,'home.html')

def predict(request):
	# if request.method=='POST':
	# 	form=ApplicationForm(request.POST)
	# 	if form.is_valid():
	probability=''
	EnglishLevel=0
	AppliedSchool=''
	Major=''
	Degree=''
	TOEFL=0
	IELTS=0
	GRE=0
	School=''
	SchoolMajor=''
	GPA=0
	column_names=joblib.load('columns.pkl')
	estimator=joblib.load('predict.pkl')

	school_list=NewResultEqual.objects.values_list('school',flat=True).distinct().exclude(school__isnull=True)
	applyschool_list=NewResultEqual.objects.values_list('appliedschool',flat=True).distinct()
	major_list=NewResultEqual.objects.values_list('major',flat=True).distinct().exclude(major__isnull=True)
	#major_list=NewResultEqual.objects.filter(appliedschool=AppliedSchool).values_list('major',flat=True).distinct().exclude(major__isnull=True)
	degree_list=NewResultEqual.objects.values_list('degree',flat=True).distinct()
	schoolmajor_list=NewResultEqual.objects.values_list('schoolmajor',flat=True).distinct().exclude(schoolmajor__isnull=True)


	# if request.method=='POST':
	# 	AppliedSchool1=request.GET.get('AppliedSchool1',False)
	# 	print (AppliedSchool1)
	# 	major_list=NewResultEqual.objects.filter(appliedschool=AppliedSchool1).values_list('major',flat=True).distinct().exclude(major__isnull=True)

	if request.method=='POST':
		# print (request.POST['AppliedSchool'])
		# print (request.POST['Major'])
		# print (request.POST['Degree'])
		# print (request.POST['TOEFL'])
		# print (request.POST['IELTS'])
		# print (request.POST['GRE'])
		# print (request.POST['School'])
		# print (request.POST['SchoolMajor'])
		# print (request.POST['GPA'])
		AppliedSchool=request.POST['AppliedSchool']
		#major_list=NewResultEqual.objects.filter(appliedschool=AppliedSchool).values_list('major',flat=True).distinct().exclude(major__isnull=True)
		Major=request.POST['Major']
		Degree=request.POST['Degree']
		TOEFL=request.POST['TOEFL']
		if TOEFL!='':
			TOEFL=float(TOEFL)
		else:
			TOEFL=0
		IELTS=request.POST['IELTS']
		if IELTS!='':
			IELTS=float(IELTS)
		else:
			IELTS=0
		if (TOEFL==0) or (IELTS==0):
			EnglishLevel=0
		if (TOEFL>0 and TOEFL<90) or (IELTS>0 and IELTS<6):
			EnglishLevel=1
		if (TOEFL>=90 and TOEFL<100) or (IELTS>=6.5 and IELTS<7):
			EnglishLevel=2
		if (TOEFL>=100 and TOEFL<110) or (IELTS>=7 and IELTS<7.5):
			EnglishLevel=3
		if (TOEFL>=110 and TOEFL<=120) or (IELTS>=7.5 and IELTS<=9):
			EnglishLevel=4

		GRE=request.POST['GRE']
		if GRE!='':
			GRE=float(GRE)
		else:
			GRE=0
		School=request.POST['School']
		SchoolMajor=request.POST['SchoolMajor']
		GPA=float(request.POST['GPA'])


		
		test=pd.DataFrame({'AppliedSchool':AppliedSchool,'Degree':Degree,'Major':Major,'SchoolMajor':SchoolMajor,'GRE':GRE,'EnglishLevel':EnglishLevel,'School':School,'GPA':GPA},index=[0])
		test1=pd.get_dummies(test)
		test2=test1.reindex(columns=column_names,fill_value=0)
		probability=float(estimator.predict_proba(test2)[:,0])*100
		probability='{0:.2f}'.format(probability)





		

	# school_list=NewResultEqual.objects.values_list('school',flat=True).distinct().exclude(school__isnull=True)
	# applyschool_list=NewResultEqual.objects.values_list('appliedschool',flat=True).distinct()
	# major_list=NewResultEqual.objects.values_list('major',flat=True).distinct().exclude(major__isnull=True)
	# #major_list=NewResultEqual.objects.filter(appliedschool=AppliedSchool).values_list('major',flat=True).distinct().exclude(major__isnull=True)
	# degree_list=NewResultEqual.objects.values_list('degree',flat=True).distinct()
	# schoolmajor_list=NewResultEqual.objects.values_list('schoolmajor',flat=True).distinct().exclude(schoolmajor__isnull=True)

	return render(request,'predict.html',{'probability':probability,'school_list':school_list,'applyschool_list':applyschool_list,'major_list':major_list,'degree_list':degree_list,'schoolmajor_list':schoolmajor_list})

def show(request):
	applyschool_list=NewResultMissing.objects.values_list('appliedschool',flat=True).distinct()
	major_list=NewResultMissing.objects.values_list('major',flat=True).distinct().exclude(major__isnull=True)
	GPA_list=''
	IELTS_list=''
	TOEFL_list=''
	GRE_list=''
	SchoolMajor_list=''
	School_list=''
	AppliedSchool_list=''
	Major_list=''
	
	GPA_list_count=''
	IELTS_list_count=''
	TOEFL_list_count=''
	GRE_list_count=''
	SchoolMajor_list_count=''
	School_list_count=''
	AppliedSchool_list_count=''
	Major_list_count=''



	if request.method=='POST':
		AppliedSchool=request.POST['AppliedSchool']
		Major=request.POST['Major']

		if (AppliedSchool!='') and (Major!=''):
			GPA_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count'))[:10]
			GPA_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count').values_list('count',flat=True))[:10]
			IELTS_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count'))[:10]
			IELTS_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count').values_list('count',flat=True))[:10]
			TOEFL_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count'))[:10]
			TOEFL_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count').values_list('count',flat=True))[:10]
			GRE_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count'))[:10]
			GRE_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count').values_list('count',flat=True))[:10]
			SchoolMajor_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count'))[:10]
			SchoolMajor_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count').values_list('count',flat=True))[:10]
			School_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count'))[:10]
			School_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool,major=Major).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count').values_list('count',flat=True))[:10]
			AppliedSchool_list=''
			Major_list=''

		if (AppliedSchool!='') and (Major==''):
			GPA_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count'))[:10]
			GPA_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count').values_list('count',flat=True))[:10]
			IELTS_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count'))[:10]
			IELTS_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count').values_list('count',flat=True))[:10]
			TOEFL_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count'))[:10]
			TOEFL_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count').values_list('count',flat=True))[:10]
			GRE_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count'))[:10]
			GRE_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count').values_list('count',flat=True))[:10]
			SchoolMajor_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count'))[:10]
			SchoolMajor_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count').values_list('count',flat=True))[:10]
			School_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count'))[:10]
			School_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count').values_list('count',flat=True))[:10]
			AppliedSchool_list=''
			AppliedSchool_list_count=''
			Major_list=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('major',flat=True).exclude(major__isnull=True).annotate(count=Count('major')).order_by('-count'))[:10]
			Major_list_count=list(NewResultMissing.objects.filter(appliedschool=AppliedSchool).values_list('major',flat=True).exclude(major__isnull=True).annotate(count=Count('major')).order_by('-count').values_list('count',flat=True))[:10]

		if (AppliedSchool=='') and (Major!=''):
			GPA_list=list(NewResultMissing.objects.filter(major=Major).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count'))[:10]
			GPA_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('gpa',flat=True).exclude(gpa__isnull=True).annotate(count=Count('gpa')).order_by('-count').values_list('count',flat=True))[:10]
			IELTS_list=list(NewResultMissing.objects.filter(major=Major).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count'))[:10]
			IELTS_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('ielts',flat=True).exclude(ielts__isnull=True).annotate(count=Count('ielts')).order_by('-count').values_list('count',flat=True))[:10]
			TOEFL_list=list(NewResultMissing.objects.filter(major=Major).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count'))[:10]
			TOEFL_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('toefl',flat=True).exclude(toefl__isnull=True).annotate(count=Count('toefl')).order_by('-count').values_list('count',flat=True))[:10]
			GRE_list=list(NewResultMissing.objects.filter(major=Major).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count'))[:10]
			GRE_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('gre',flat=True).exclude(gre__isnull=True).annotate(count=Count('gre')).order_by('-count').values_list('count',flat=True))[:10]
			SchoolMajor_list=list(NewResultMissing.objects.filter(major=Major).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count'))[:10]
			SchoolMajor_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('schoolmajor',flat=True).exclude(schoolmajor__isnull=True).annotate(count=Count('schoolmajor')).order_by('-count').values_list('count',flat=True))[:10]
			School_list=list(NewResultMissing.objects.filter(major=Major).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count'))[:10]
			School_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('school',flat=True).exclude(school__isnull=True).annotate(count=Count('school')).order_by('-count').values_list('count',flat=True))[:10]
			AppliedSchool_list=list(NewResultMissing.objects.filter(major=Major).values_list('appliedschool',flat=True).exclude(appliedschool__isnull=True).annotate(count=Count('appliedschool')).order_by('-count'))[:10]		
			AppliedSchool_list_count=list(NewResultMissing.objects.filter(major=Major).values_list('appliedschool',flat=True).exclude(appliedschool__isnull=True).annotate(count=Count('appliedschool')).order_by('-count').values_list('count',flat=True))[:10]
			Major_list=''
			Major_list_count=''


		# print (json.dumps(GPA_list,ensure_ascii=False))
		# print (json.dumps(GPA_list_count,ensure_ascii=False))

		# print (json.dumps(IELTS_list,ensure_ascii=False))
		# print (json.dumps(IELTS_list_count,ensure_ascii=False))

		# print (json.dumps(TOEFL_list,ensure_ascii=False))
		# print (json.dumps(TOEFL_list_count,ensure_ascii=False))

		# print (json.dumps(GRE_list,ensure_ascii=False))
		# print (json.dumps(GRE_list_count,ensure_ascii=False))

		# print (json.dumps(SchoolMajor_list,ensure_ascii=False))
		# print (json.dumps(SchoolMajor_list_count,ensure_ascii=False))

		# print (json.dumps(School_list,ensure_ascii=False))
		# print (json.dumps(School_list_count,ensure_ascii=False))

		# print (json.dumps(AppliedSchool_list,ensure_ascii=False))
		# print (json.dumps(AppliedSchool_list_count,ensure_ascii=False))

		# print (json.dumps(Major_list,ensure_ascii=False))
		# print (json.dumps(Major_list_count,ensure_ascii=False))






	
	return render(request,'show.html',{'applyschool_list':applyschool_list,'major_list':major_list,
		'GPA_list':json.dumps(GPA_list,ensure_ascii=False),
		'GPA_list_count':json.dumps(GPA_list_count,ensure_ascii=False),

		'IELTS_list':json.dumps(IELTS_list,ensure_ascii=False),
		'IELTS_list_count':json.dumps(IELTS_list_count,ensure_ascii=False),

		'TOEFL_list':json.dumps(TOEFL_list,ensure_ascii=False),
		'TOEFL_list_count':json.dumps(TOEFL_list_count,ensure_ascii=False),

		'GRE_list':json.dumps(GRE_list,ensure_ascii=False),
		'GRE_list_count':json.dumps(GRE_list_count,ensure_ascii=False),

		'SchoolMajor_list':json.dumps(SchoolMajor_list,ensure_ascii=False),
		'SchoolMajor_list_count':json.dumps(SchoolMajor_list_count,ensure_ascii=False),

		'School_list':json.dumps(School_list,ensure_ascii=False),
		'School_list_count':json.dumps(School_list_count,ensure_ascii=False),

		'AppliedSchool_list':json.dumps(AppliedSchool_list,ensure_ascii=False),
		'AppliedSchool_list_count':json.dumps(AppliedSchool_list_count,ensure_ascii=False),

		'Major_list':json.dumps(Major_list,ensure_ascii=False),
		'Major_list_count':json.dumps(Major_list_count,ensure_ascii=False)
		})

# def predict_result(request):
# 	if request.method=='POST':
# 		print ("it's a test")
# 		print (request.POST['AppliedSchool'])
# 		print (request.POST['Major'])
# 		print (request.POST['Degree'])
# 		print (request.POST['TOEFL'])
# 		print (request.POST['IELTS'])
# 		print (request.POST['GRE'])
# 		print (request.POST['School'])
# 		print (request.POST['SchoolMajor'])
# 		print (request.POST['GPA'])

# 		return (HttpResponse('test success'))
# 	else:
# 		return (HttpResponse('test fail'))




# def application_edit(request):
# 	cascade_select_list=[('province')]
# 	if request.method=='POST':
# 		form=ApplicationForm(request.POST,request.FILES,instance=request.profile)
# 		if form.is_valid():
# 			new_profile=form.save()
# 			request.user.message_set.create()







