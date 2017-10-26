from django.http import HttpResponse
from predict.models import NewResultMissing

def testdb_getall(request):
	test1=Application_Info_Missing(AppliedSchool='爱丁堡大学',Degree='硕士',Major='Artificial Intelligence',GPA=90,Result='录取',School='211',SchoolMajor='计算机科学')
	test1.save()
	return HttpResponse('<p>数据添加成功！</p>')
	# test1=Application_Info_Missing.objects.all()
	# test1.delete()
	# return HttpResponse('<p>删除成功</p>')
	# test1=Application_Info_Missing.objects.all()
