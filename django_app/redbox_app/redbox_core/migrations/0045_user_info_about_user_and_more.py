# Generated by Django 5.1.1 on 2024-09-26 11:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0044_aisettings_rag_gauss_scale_decay_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='info_about_user',
            field=models.CharField(blank=True, help_text='user entered info from profile overlay', null=True),
        ),
        migrations.AddField(
            model_name='user',
            name='redbox_response_preferences',
            field=models.CharField(blank=True, help_text='user entered info from profile overlay, to be used in custom prompt', null=True),
        ),
    ]
