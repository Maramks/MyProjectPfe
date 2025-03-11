from django.db import models

class CarteCINTemp(models.Model):
    numero_cin = models.CharField(max_length=20, unique=True, db_column='رقم_البطاقة')  # Numéro de CIN
    nom = models.CharField(max_length=100, db_column='الاسم')  # Nom (correspond à الاسم en arabe)
    prenom = models.CharField(max_length=100, db_column='اللقب')  # Prénom (correspond à اللقب en arabe)
    date_naissance = models.DateField(db_column='تاريخ_الميلاد')  # Date de naissance (correspond à تاريخ_الميلاد en arabe)
    adresse = models.TextField(db_column='العنوان')  # Adresse (correspond à العنوان en arabe)
    image_cin = models.BinaryField(db_column='صورة_البطاقة')  # Image du recto de la CIN (correspond à صورة_البطاقة en arabe)
    selfie = models.BinaryField(db_column='صورة_الوجه')  # Selfie de l'utilisateur (correspond à صورة_الوجه en arabe)
    code_barre = models.CharField(max_length=50, db_column='الرمز_الشريطي')  # Code-barres extrait du verso (correspond à الرمز_الشريطي en arabe)
    verification_reussie = models.BooleanField(default=False, db_column='تم_التحقق')  # Vérification réussie ou non (correspond à تم_التحقق en arabe)
    date_ajout = models.DateTimeField(auto_now_add=True, db_column='تاريخ_الإدخال')  # Date d'ajout (correspond à تاريخ_الإدخال en arabe)

    def __str__(self):
        return self.numero_cin


class CarteCINValide(models.Model):
    numero_cin = models.CharField(max_length=20, unique=True, db_column='رقم_البطاقة')  # Numéro de CIN
    nom = models.CharField(max_length=100, db_column='الاسم')  # Nom (correspond à الاسم en arabe)
    prenom = models.CharField(max_length=100, db_column='اللقب')  # Prénom (correspond à اللقب en arabe)
    date_naissance = models.DateField(db_column='تاريخ_الميلاد')  # Date de naissance (correspond à تاريخ_الميلاد en arabe)
    adresse = models.TextField(db_column='العنوان')  # Adresse (correspond à العنوان en arabe)
    image_cin = models.BinaryField(db_column='صورة_البطاقة')  # Image du recto de la CIN (correspond à صورة_البطاقة en arabe)
    date_ajout = models.DateTimeField(auto_now_add=True, db_column='تاريخ_الإدخال')  # Date d'ajout (correspond à تاريخ_الإدخال en arabe)

    def __str__(self):
        return self.numero_cin


class Utilisateur(models.Model):
    numero_cin = models.CharField(max_length=20, unique=True, db_column='رقم_البطاقة')  # Numéro de CIN
    email = models.EmailField(unique=True, db_column='البريد_الإلكتروني')  # Email
    mot_de_passe = models.CharField(max_length=255, db_column='كلمة_المرور')  # Mot de passe
    date_enregistrement = models.DateTimeField(auto_now_add=True, db_column='تاريخ_التسجيل')  # Date d'enregistrement

    # Relation avec la table cartes_cin_valides
    carte_cin = models.ForeignKey(CarteCINValide, on_delete=models.CASCADE, db_column='رقم_البطاقة_utilisateur')  # Utilisation d'un nom de colonne distinct

    def __str__(self):
        return f"Utilisateur: {self.email}"
    
class Agence(models.Model):
    nom = models.CharField(max_length=255)
    adresse = models.TextField()
    telephone = models.CharField(max_length=20)
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return self.nom
    


class Admin(models.Model):
    matricule = models.IntegerField(unique=True)  # Matricule de l'administrateur (champ entier)
    email = models.EmailField(null=True)  # Email de l'administrateur
    numéro_téléphone = models.CharField(max_length=20, null=True)  # Numéro de téléphone
    nom = models.CharField(max_length=100, null=True)  # Nom de l'administrateur
    prénom = models.CharField(max_length=100, null=True)  # Prénom de l'administrateur
    
    def __str__(self):
        return f"Admin: {self.matricule} - {self.nom} {self.prénom}"
