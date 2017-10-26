# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from __future__ import unicode_literals

from django.db import models
from django import forms
from smart_selects.db_fields import ChainedManyToManyField


class NewResult(models.Model):
    index = models.BigIntegerField(blank=True,primary_key=True)
    applied_school = models.TextField(db_column='Applied_School', blank=True, null=True)  # Field name made lowercase.
    degree = models.TextField(db_column='Degree', blank=True, null=True)  # Field name made lowercase.
    major = models.TextField(db_column='Major', blank=True, null=True)  # Field name made lowercase.
    result = models.TextField(db_column='Result', blank=True, null=True)  # Field name made lowercase.
    year = models.TextField(db_column='Year', blank=True, null=True)  # Field name made lowercase.
    toefl = models.TextField(db_column='TOEFL', blank=True, null=True)  # Field name made lowercase.
    ielts = models.TextField(db_column='IELTS', blank=True, null=True)  # Field name made lowercase.
    gre = models.TextField(db_column='GRE', blank=True, null=True)  # Field name made lowercase.
    school = models.TextField(db_column='School', blank=True, null=True)  # Field name made lowercase.
    school_major = models.TextField(db_column='School_Major', blank=True, null=True)  # Field name made lowercase.
    gpa = models.TextField(db_column='GPA', blank=True, null=True)  # Field name made lowercase.
    post_school = models.TextField(db_column='Post_School', blank=True, null=True)  # Field name made lowercase.
    post_major = models.TextField(db_column='Post_Major', blank=True, null=True)  # Field name made lowercase.
    post_gpa = models.TextField(db_column='Post_GPA', blank=True, null=True)  # Field name made lowercase.
    other_info = models.TextField(db_column='Other_Info', blank=True, null=True)  # Field name made lowercase.
    def __str__(self):
        return "%d 学校:%s 申请专业:%s GPA:%.2f" %(self.index,self.applied_school,self.major,self.gpa)

    class Meta:
        managed = False
        db_table = 'new_result'


class NewResultEqual(models.Model):
    index = models.BigIntegerField(blank=True,primary_key=True)
    appliedschool = models.TextField(db_column='AppliedSchool', blank=True, null=True)  # Field name made lowercase.
    degree = models.TextField(db_column='Degree', blank=True, null=True)  # Field name made lowercase.
    major = models.TextField(db_column='Major', blank=True, null=True)  # Field name made lowercase.
    result = models.TextField(db_column='Result', blank=True, null=True)  # Field name made lowercase.
    year = models.BigIntegerField(db_column='Year', blank=True, null=True)  # Field name made lowercase.
    englishlevel = models.BigIntegerField(db_column='EnglishLevel', blank=True, null=True)  # Field name made lowercase.
    toefl = models.FloatField(db_column='TOEFL', blank=True, null=True)  # Field name made lowercase.
    ielts = models.FloatField(db_column='IELTS', blank=True, null=True)  # Field name made lowercase.
    gre = models.BigIntegerField(db_column='GRE', blank=True, null=True)  # Field name made lowercase.
    school = models.TextField(db_column='School', blank=True, null=True)  # Field name made lowercase.
    schoolmajor = models.TextField(db_column='SchoolMajor', blank=True, null=True)  # Field name made lowercase.
    gpa = models.FloatField(db_column='GPA', blank=True, null=True)  # Field name made lowercase.
    postschool = models.TextField(db_column='PostSchool', blank=True, null=True)  # Field name made lowercase.
    postmajor = models.TextField(db_column='PostMajor', blank=True, null=True)  # Field name made lowercase.
    postgpa = models.TextField(db_column='PostGPA', blank=True, null=True)  # Field name made lowercase.
    other_info = models.TextField(db_column='Other_Info', blank=True, null=True)  # Field name made lowercase.
    def __str__(self):
        return "%d 学校:%s 申请专业:%s GPA:%.2f" %(self.index,self.appliedschool,self.major,self.gpa)
    class Meta:
        managed = False
        db_table = 'new_result_equal'


class NewResultMissing(models.Model):
    index = models.BigIntegerField(blank=True,primary_key=True)
    appliedschool = models.TextField(db_column='AppliedSchool', blank=True, null=True)  # Field name made lowercase.
    degree = models.TextField(db_column='Degree', blank=True, null=True)  # Field name made lowercase.
    major = models.TextField(db_column='Major', blank=True, null=True)  # Field name made lowercase.
    result = models.TextField(db_column='Result', blank=True, null=True)  # Field name made lowercase.
    year = models.BigIntegerField(db_column='Year', blank=True, null=True)  # Field name made lowercase.
    englishlevel = models.BigIntegerField(db_column='EnglishLevel', blank=True, null=True)  # Field name made lowercase.
    toefl = models.FloatField(db_column='TOEFL', blank=True, null=True)  # Field name made lowercase.
    ielts = models.FloatField(db_column='IELTS', blank=True, null=True)  # Field name made lowercase.
    gre = models.BigIntegerField(db_column='GRE', blank=True, null=True)  # Field name made lowercase.
    school = models.TextField(db_column='School', blank=True, null=True)  # Field name made lowercase.
    schoolmajor = models.TextField(db_column='SchoolMajor', blank=True, null=True)  # Field name made lowercase.
    gpa = models.FloatField(db_column='GPA', blank=True, null=True)  # Field name made lowercase.
    postschool = models.TextField(db_column='PostSchool', blank=True, null=True)  # Field name made lowercase.
    postmajor = models.TextField(db_column='PostMajor', blank=True, null=True)  # Field name made lowercase.
    postgpa = models.TextField(db_column='PostGPA', blank=True, null=True)  # Field name made lowercase.
    other_info = models.TextField(db_column='Other_Info', blank=True, null=True)  # Field name made lowercase.
    def __str__(self):
        return "%d 学校:%s 申请专业:%s GPA:%.2f" %(self.index,self.appliedschool,self.major,self.gpa)
    class Meta:
        managed = False
        db_table = 'new_result_missing'

# class ApplicationForm(forms.ModelForm):
#     class Meta:
#         model=NewResultEqual
#         field=('appliedschool','major')

# class AppliedSchool(models.Model):
#     name=models.CharField(max_length=255)

# class Major(models.Model):
#     name=models.CharField(max_length=255)
#     appliedschool=models.ForeignKey(AppliedSchool)
    


