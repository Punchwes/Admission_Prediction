from django.contrib import admin


# Register your models here.
from predict.models import NewResultEqual,NewResultMissing 

# Register your models here.
admin.site.register([NewResultMissing,NewResultEqual])
