# Generated by Django 3.2.13 on 2022-05-10 01:56

from django.db import migrations, models
import home.utils


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_element_delete_element_oo'),
    ]

    operations = [
        migrations.AlterField(
            model_name='element',
            name='element_Img',
            field=models.ImageField(storage=home.utils.MyFileStorage(), upload_to='home/static/images/'),
        ),
    ]