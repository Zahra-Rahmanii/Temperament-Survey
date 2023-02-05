from django import forms
from django.forms import ModelForm,RadioSelect,Select
from .models import Temper
from TMtest.choices import *


class DataForm(ModelForm):
    class Meta:
        model = Temper
        fields = ['name','weight','height','age','gender','skin_color','nail_color','iris_color','lip_shape','lip_color','nose_shape','hair_color','white_hair','hairless_part','hand_vein','abdobinal_form','tooth_color','sleep','urine_color','stool_color','body_shape','talking','mouth_shape','mood','hair','your_feeling','introvert','grudge','bad_memories','unwanted_hair','skin_diseas','dry_lip','sweatimg','sparse_hair','hair_loss','itchy_head','dandruff','dry_mouth','frequent_urination','cold_hand','curiouse']
        widgets={
            'gender': Select(),
            'skin_color':RadioSelect(),
            'nail_color':RadioSelect(),
            'iris_color':RadioSelect(),
            'lip_shape':RadioSelect(),
            'lip_color':forms.RadioSelect(choices=lip_color),
            'nose_shape':forms.RadioSelect(choices=nose_shape),
            'hair_color':forms.RadioSelect(choices=hair_color),
            'white_hair':forms.RadioSelect(choices=white_hair),
            'hairless_part':forms.RadioSelect(choices=hairless_part),
            'hand_vein':forms.RadioSelect(choices=hand_vein),
            'abdobinal_form':forms.RadioSelect(choices=abdobinal_form),
            'tooth_color':forms.RadioSelect(choices=tooth_color),
            'sleep':forms.RadioSelect(choices=sleep),
            'urine_color':forms.RadioSelect(choices=urine_color),
            'stool_color':forms.RadioSelect(),
            'body_shape':forms.RadioSelect(),
            'talking':forms.RadioSelect(),
            'mouth_shape':forms.RadioSelect(),
            'mood':forms.RadioSelect(),
            'hair':forms.RadioSelect(),
            'your_feeling':forms.RadioSelect(),
            'introvert':forms.RadioSelect(),
            'grudge':forms.RadioSelect(),
            'mouth_shape':forms.RadioSelect(),
            'bad_memories':forms.RadioSelect(),
            'unwanted_hair':forms.RadioSelect(),
            'skin_diseas':forms.RadioSelect(),
            'dry_lip':forms.RadioSelect(),
            'sweatimg':forms.RadioSelect(),
            'sparse_hair':forms.RadioSelect(),
            'hair_loss':forms.RadioSelect(),
            'itchy_head':forms.RadioSelect(),
            'dandruff':forms.RadioSelect(),
            'dry_mouth':forms.RadioSelect(),
            'frequent_urination':forms.RadioSelect(),
            'cold_hand':forms.RadioSelect(),
            'curiouse':forms.RadioSelect()
        }
    