from django.db import models
from .choices import *
from sklearn.tree import DecisionTreeClassifier
import joblib

# Create your models here.
class Temper(models.Model):
    name=models.CharField(max_length=200,null=True,verbose_name='نام و نام خانوادگی:')
    weight=models.PositiveBigIntegerField(verbose_name='وزن')
    height=models.PositiveBigIntegerField(verbose_name='قد')
    age=models.PositiveBigIntegerField(verbose_name='سن:')
    gender=models.IntegerField(choices=Gender,verbose_name='جنسیت:')
    skin_color=models.IntegerField(choices=Skin_Color,null=True,verbose_name='پوست صورت شما به چه رنگی است؟')
    nail_color=models.IntegerField(choices=Nail_Color,null=True,verbose_name='ناخن های شما به چه رنگی است؟')
    iris_color=models.IntegerField(choices=Iris_Color,null=True,verbose_name='عنبیه چشم شما به چه رنگ است؟')
    lip_shape=models.IntegerField(choices=lip_shape,null=True,verbose_name='فرم لب های شما به چه صورت است؟')
    lip_color=models.IntegerField(choices=lip_color,null=True,verbose_name='معمولا لب های شما به چه رنگی است؟')
    nose_shape=models.IntegerField(choices=nose_shape,null=True,verbose_name='فرم بینی شما به چه صورت است؟')
    hair_color=models.IntegerField(choices=hair_color,null=True,verbose_name='موهای شما به چه رنگی است؟')
    white_hair=models.IntegerField(choices=white_hair,null=True,verbose_name='چند درصد از موهای شما سفید شده است؟')
    hairless_part=models.IntegerField(choices=hairless_part,null=True,verbose_name='کدام قسمت از موهای سرتان ریزش بیشتر ی دارد؟')
    hand_vein=models.IntegerField(choices=hand_vein,null=True,verbose_name='کدام مورد درمورد رگ های دستتان صدق می کند؟')
    abdobinal_form=models.IntegerField(choices=abdobinal_form,null=True,verbose_name='فزم شکم شما به چه صورت است؟')
    tooth_color=models.IntegerField(choices=tooth_color,null=True,verbose_name='دندان های شما معمولا چه رنگی است؟')
    sleep=models.IntegerField(choices=sleep,null=True,verbose_name='کدام مورد در مورد خواب شما صدق می کند؟')
    urine_color=models.IntegerField(choices=urine_color,null=True,verbose_name='ادرار شما در طول روز به چه رنگی است؟')
    stool_color=models.IntegerField(choices=Stool_Color,null=True,verbose_name='معمولا مدفوع شما به چه رنگی است؟')
    body_shape=models.IntegerField(choices=Body_Shape,null=True,verbose_name='فرم فیکل شما به چه شکلی است؟')
    talking=models.IntegerField(choices=Talking,null=True,verbose_name='نحوه حرف زدن شما به چه صورت است؟')
    mouth_shape=models.IntegerField(choices=Mouth_shape,null=True,verbose_name='فرم دهان شما به چه صورت است؟')
    mood=models.IntegerField(choices=Mood,null=True,verbose_name='کدام مورد در مورد شما درست است؟')
    hair=models.IntegerField(choices=Hair,null=True,verbose_name='موهای شما به چه صورت است؟')
    your_feeling=models.IntegerField(choices=Your_Feeling,null=True,verbose_name='در بیشتر فصول سال معمولا شما احساس گرما می کنید یا سرما؟')
    introvert=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا شما فردی درون گرا هستید؟')
    grudge=models.IntegerField(choices=check_yes,null=True,verbose_name='اگر کسی در حق شما بدی انجام دهد بدی که ئذ حقتان کرده را بعد از مدتی فراموش می کنید؟')
    bad_memories=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا خاطرات بد گذشته را همیشه به یاد می آورید؟')
    unwanted_hair=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا موهای زائد بدن شما زیاد و کلفت است؟')
    skin_diseas=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا از بیماری پوستی رنج می برید')
    dry_lip=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا معمولا لب هایتان خشک می شود؟')
    sweatimg=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا تعریق شما زیاد است؟')
    sparse_hair=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا معمولا موهایتان کم پشت است؟')
    hair_loss=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا شما از ریزش موهایتان رنج می برید؟')
    itchy_head=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا کف سر شما دچار خارش می شود؟')
    dandruff=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا معمولا موهایتان شوره دارد')
    dry_mouth=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا معمولا دهانتان خشک است و آب زیادی مصرف می کنید؟')
    frequent_urination=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا تکرر ادرار دارید؟')
    cold_hand=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا معمولا هنگامی که کسی دست هایتان را لمس می کند احساس سرما می کند؟')
    curiouse=models.IntegerField(choices=check_yes,null=True,verbose_name='آیا شما فردی کنجکاو و ریزبین هستید؟')
    dry=models.CharField(max_length=10,blank=True)
    cold=models.CharField(max_length=10,blank=True)
    date = models.DateTimeField(auto_now_add=True,blank=True)

    def save(self, *args, **kwargs):
        bmi=self.weight/(self.height**2)
        if bmi < 18:
            bmi=1
        elif bmi < 25 and bmi > 18:
            bmi=2
        elif bmi < 30 and bmi > 25 :
            bmi=3
        else:
            bmi=4
        if self.age < 30 :
            age=1
        elif  self.age < 55 and self.age >=  30:
            age=2
        else:
            age=3
        ml_model1 = joblib.load('ml_model/model_dry.joblib')
        ml_model2=joblib.load('ml_model/model_cold.joblib')
        print([bmi, self.your_feeling, self.skin_color,self.mood,self.iris_color,self.hand_vein,self.stool_color,self.lip_shape,self.lip_color,self.nose_shape,self.hair_color,self.white_hair,self.hairless_part,self.abdobinal_form,self.tooth_color,self.sleep,self.mouth_shape,self.urine_color,self.talking,self.body_shape,self.nail_color,self.hair,self.introvert,self.grudge,self.bad_memories,self.unwanted_hair,self.skin_diseas,self.dry_lip,self.sweatimg,self.sparse_hair,self.hair_loss,self.itchy_head,self.dandruff,self.dry_mouth,self.frequent_urination,self.cold_hand,self.curiouse,age,self.gender])
        self.dry = ml_model1.predict(
            [[bmi, self.your_feeling, self.skin_color,self.mood,self.iris_color,self.hand_vein,self.stool_color,self.lip_shape,self.lip_color,self.nose_shape,self.hair_color,self.white_hair,self.hairless_part,self.abdobinal_form,self.tooth_color,self.sleep,self.mouth_shape,self.urine_color,self.talking,self.body_shape,self.nail_color,self.hair,self.introvert,self.grudge,self.bad_memories,self.unwanted_hair,self.skin_diseas,self.dry_lip,self.sweatimg,self.sparse_hair,self.hair_loss,self.itchy_head,self.dandruff,self.dry_mouth,self.frequent_urination,self.cold_hand,self.curiouse,age,self.gender]])
        self.cold = ml_model2.predict(
            [[bmi, self.your_feeling, self.skin_color,self.mood,self.iris_color,self.hand_vein,self.stool_color,self.lip_shape,self.lip_color,self.nose_shape,self.hair_color,self.white_hair,self.hairless_part,self.abdobinal_form,self.tooth_color,self.sleep,self.mouth_shape,self.urine_color,self.talking,self.body_shape,self.nail_color,self.hair,self.introvert,self.grudge,self.bad_memories,self.unwanted_hair,self.skin_diseas,self.dry_lip,self.sweatimg,self.sparse_hair,self.hair_loss,self.itchy_head,self.dandruff,self.dry_mouth,self.frequent_urination,self.cold_hand,self.curiouse,age,self.gender]])
        print(self.dry)
        return super().save(*args, *kwargs)


    class Meta:
        ordering = ['-date']

    def __str__(self):
        return self.name
    