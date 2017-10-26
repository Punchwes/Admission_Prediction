from django import forms
from models import NewResultEqual,NewResultMissing

#class application_form(forms.ModelForm):
# appliedschools=NewResultEqual.objects.values_list('school',flat=True).distinct().exclude(school__isnull=True)
# appliedschool_choice=[]
# for appliedschool in appliedschools:
# 	appliedschool_choice.append(appliedschool)

# class myForm(forms.ModelForm):
# 	appliedschool=forms.TypedChoiceField