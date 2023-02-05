# Generated by Django 4.1.5 on 2023-02-04 23:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Temper',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200, null=True, verbose_name='نام و نام خانوادگی:')),
                ('weight', models.PositiveBigIntegerField(verbose_name='وزن')),
                ('height', models.PositiveBigIntegerField(verbose_name='قد')),
                ('age', models.PositiveBigIntegerField(verbose_name='سن:')),
                ('gender', models.IntegerField(choices=[(1, 'famale'), (0, 'male')], verbose_name='جنسیت:')),
                ('skin_color', models.IntegerField(choices=[(1, 'سفید گلگون'), (2, 'گندمی متمایل به زرد'), (3, 'سفید بی حال'), (4, 'تیره متمایل به سبزه')], null=True, verbose_name='پوست صورت شما به چه رنگی است؟')),
                ('nail_color', models.IntegerField(choices=[(1, 'صورتی متمایل به بنفش'), (2, 'متمایل به زرد'), (3, 'رو به سفید'), (4, 'متمایل به خاکستری')], null=True, verbose_name='ناخن های شما به چه رنگی است؟')),
                ('iris_color', models.IntegerField(choices=[(1, 'رنگی (آبی،سبزو ..)'), (2, 'قهوه ای روشن'), (3, 'قهوه ای تیره'), (4, 'سیاه')], null=True, verbose_name='عنبیه چشم شما به چه رنگ است؟')),
                ('lip_shape', models.IntegerField(choices=[(4, 'قلوه ای'), (6, 'باریک و نازک'), (7, 'معمولی')], null=True, verbose_name='فرم لب های شما به چه صورت است؟')),
                ('lip_color', models.IntegerField(choices=[(1, 'صورتی معمولی'), (2, 'متمایل به قرمز'), (3, 'صورتی متمایل به سفید'), (4, 'تیره')], null=True, verbose_name='معمولا لب های شما به چه رنگی است؟')),
                ('nose_shape', models.IntegerField(choices=[(4, 'بینی عقابی'), (4, 'بینی کوچک با پره های نازک'), (0, 'متوسط'), (2, 'بینی بزرگ')], null=True, verbose_name='فرم بینی شما به چه صورت است؟')),
                ('hair_color', models.IntegerField(choices=[(2, 'بور و رنگی'), (8, 'قهوه ای'), (4, 'مشکی(پرکلاغی)')], null=True, verbose_name='موهای شما به چه رنگی است؟')),
                ('white_hair', models.IntegerField(choices=[(1, 'زیر 30 درصد'), (2, 'بین 30 تا 60 درصد'), (3, 'بالای 60 درصد'), (0, 'اصلا تا حالا موی سفید نداشته اید')], null=True, verbose_name='چند درصد از موهای شما سفید شده است؟')),
                ('hairless_part', models.IntegerField(choices=[(1, 'پشت سر'), (2, 'جلوی سر'), (4, 'کف سر'), (0, 'هیچکدام')], null=True, verbose_name='کدام قسمت از موهای سرتان ریزش بیشتر ی دارد؟')),
                ('hand_vein', models.IntegerField(choices=[(5, 'رگ های برحسته و کلفت'), (4, 'رگ های برجسته و نازک'), (3, 'پیدا نیست')], null=True, verbose_name='کدام مورد درمورد رگ های دستتان صدق می کند؟')),
                ('abdobinal_form', models.IntegerField(choices=[(7, 'بزرگی خود شکم'), (8, 'شکم ندارید'), (0, 'بزرگی دور شکم')], null=True, verbose_name='فزم شکم شما به چه صورت است؟')),
                ('tooth_color', models.IntegerField(choices=[(2, 'متمایل به زرد'), (4, 'کملا سفید')], null=True, verbose_name='دندان های شما معمولا چه رنگی است؟')),
                ('sleep', models.IntegerField(choices=[(2, 'معمول سر شب خوابتان می گیرد'), (4, 'شب ها معمولا تا دیر وقت خوابتان نمی گیرد'), (0, 'در طول روز هم علاوه بر شب می خوابید')], null=True, verbose_name='کدام مورد در مورد خواب شما صدق می کند؟')),
                ('urine_color', models.IntegerField(choices=[(4, 'رنگ آن کدر و تیره است'), (7, 'بی رنگ و مایل به سفید'), (2, 'زرد روشن')], null=True, verbose_name='ادرار شما در طول روز به چه رنگی است؟')),
                ('stool_color', models.IntegerField(choices=[(1, 'قهوه ای'), (2, ' قهوه ای متمایل به زرد'), (3, ' متمایل به سیاه'), (0, 'هیچکدام')], null=True, verbose_name='معمولا مدفوع شما به چه رنگی است؟')),
                ('body_shape', models.IntegerField(choices=[(1, 'چهارشانه(استخوانبندی درشت) و قد بلند'), (2, 'جهارشانه و متوسط'), (3, 'لاغر و قد بلند'), (4, 'لاغر و متوسط یا کوتاه'), (6, 'جاق و قد بلند'), (7, 'چاق و قد متوسط یا کوتاه')], null=True, verbose_name='فرم فیکل شما به چه شکلی است؟')),
                ('talking', models.IntegerField(choices=[(2, 'معمولا تند تند حرف می زنید '), (4, 'معمولا آرام حرف می زنید'), (0, 'نه آرام حرف می زنید و نه تند تند')], null=True, verbose_name='نحوه حرف زدن شما به چه صورت است؟')),
                ('mouth_shape', models.IntegerField(choices=[(1, 'دهان بززرگ'), (1, 'دهان کوچک'), (1, 'معمولی')], null=True, verbose_name='فرم دهان شما به چه صورت است؟')),
                ('mood', models.IntegerField(choices=[(6, 'معمولا خیلی زود عصبانی می شوید ولی زود هم آرام می شوید'), (5, ' معمولا خیلی دیر عصبانی می شوید و خیلی دیر آرام می شوید')], null=True, verbose_name='کدام مورد در مورد شما درست است؟')),
                ('hair', models.IntegerField(choices=[(2, 'لخت و نازک'), (4, 'لخت و کلفت '), (2, 'فر ریز'), (4, 'فر درشت')], null=True, verbose_name='موهای شما به چه صورت است؟')),
                ('your_feeling', models.IntegerField(choices=[(6, 'در بیستر فصول سال احساس سرما می کنید(سرمایی هستید)'), (5, 'در بیشتر فصول سال احساس گرما می کنید(گرمایی هستید).')], null=True, verbose_name='در بیشتر فصول سال معمولا شما احساس گرما می کنید یا سرما؟')),
                ('introvert', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا شما فردی درون گرا هستید؟')),
                ('grudge', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='اگر کسی در حق شما بدی انجام دهد بدی که ئذ حقتان کرده را بعد از مدتی فراموش می کنید؟')),
                ('bad_memories', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا خاطرات بد گذشته را همیشه به یاد می آورید؟')),
                ('unwanted_hair', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا موهای زائد بدن شما زیاد و کلفت است؟')),
                ('skin_diseas', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا از بیماری پوستی رنج می برید')),
                ('dry_lip', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا معمولا لب هایتان خشک می شود؟')),
                ('sweatimg', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا تعریق شما زیاد است؟')),
                ('sparse_hair', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا معمولا موهایتان کم پشت است؟')),
                ('hair_loss', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا شما از ریزش موهایتان رنج می برید؟')),
                ('itchy_head', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا کف سر شما دچار خارش می شود؟')),
                ('dandruff', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا معمولا موهایتان شوره دارد')),
                ('dry_mouth', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا معمولا دهانتان خشک است و آب زیادی مصرف می کنید؟')),
                ('frequent_urination', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا تکرر ادرار دارید؟')),
                ('cold_hand', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا معمولا هنگامی که کسی دست هایتان را لمس می کند احساس سرما می کند؟')),
                ('curiouse', models.IntegerField(choices=[(1, 'بله'), (0, 'خیر')], null=True, verbose_name='آیا شما فردی کنجکاو و ریزبین هستید؟')),
                ('dry', models.CharField(blank=True, max_length=10)),
                ('cold', models.CharField(blank=True, max_length=10)),
                ('date', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['-date'],
            },
        ),
    ]
