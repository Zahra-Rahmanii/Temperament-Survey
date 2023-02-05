from django.contrib import admin
from .models import Temper

# Register your models here.


class DataAdmin(admin.ModelAdmin):
    list_display = ('name', 'age', 'height', 'gender', 'dry','cold')


admin.site.register(Temper, DataAdmin)