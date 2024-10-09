# Generated by Django 5.1.1 on 2024-09-24 12:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0043_chat_feedback_achieved_chat_feedback_improved_work_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='aisettings',
            name='rag_gauss_scale_decay',
            field=models.FloatField(default=0.5),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='rag_gauss_scale_max',
            field=models.FloatField(default=2.0),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='rag_gauss_scale_min',
            field=models.FloatField(default=1.1),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='rag_gauss_scale_size',
            field=models.PositiveIntegerField(default=3),
        ),
    ]
