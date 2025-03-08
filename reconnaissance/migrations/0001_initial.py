# Generated by Django 5.1.7 on 2025-03-07 21:13

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CarteCINTemp',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero_cin', models.CharField(db_column='رقم_البطاقة', max_length=20, unique=True)),
                ('nom', models.CharField(db_column='الاسم', max_length=100)),
                ('prenom', models.CharField(db_column='اللقب', max_length=100)),
                ('date_naissance', models.DateField(db_column='تاريخ_الميلاد')),
                ('adresse', models.TextField(db_column='العنوان')),
                ('image_cin', models.BinaryField(db_column='صورة_البطاقة')),
                ('selfie', models.BinaryField(db_column='صورة_الوجه')),
                ('code_barre', models.CharField(db_column='الرمز_الشريطي', max_length=50)),
                ('verification_reussie', models.BooleanField(db_column='تم_التحقق', default=False)),
                ('date_ajout', models.DateTimeField(auto_now_add=True, db_column='تاريخ_الإدخال')),
            ],
        ),
        migrations.CreateModel(
            name='CarteCINValide',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero_cin', models.CharField(db_column='رقم_البطاقة', max_length=20, unique=True)),
                ('nom', models.CharField(db_column='الاسم', max_length=100)),
                ('prenom', models.CharField(db_column='اللقب', max_length=100)),
                ('date_naissance', models.DateField(db_column='تاريخ_الميلاد')),
                ('adresse', models.TextField(db_column='العنوان')),
                ('image_cin', models.BinaryField(db_column='صورة_البطاقة')),
                ('date_ajout', models.DateTimeField(auto_now_add=True, db_column='تاريخ_الإدخال')),
            ],
        ),
        migrations.CreateModel(
            name='Utilisateur',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numero_cin', models.CharField(db_column='رقم_البطاقة', max_length=20, unique=True)),
                ('email', models.EmailField(db_column='البريد_الإلكتروني', max_length=254, unique=True)),
                ('mot_de_passe', models.CharField(db_column='كلمة_المرور', max_length=255)),
                ('date_enregistrement', models.DateTimeField(auto_now_add=True, db_column='تاريخ_التسجيل')),
                ('carte_cin', models.ForeignKey(db_column='رقم_البطاقة_utilisateur', on_delete=django.db.models.deletion.CASCADE, to='reconnaissance.cartecinvalide')),
            ],
        ),
    ]
