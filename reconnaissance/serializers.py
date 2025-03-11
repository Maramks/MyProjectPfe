# serializers.py
#Les sérialiseurs dans Django REST Framework transforment les objets du modèle en formats lisibles pour les API (comme JSON).
from rest_framework import serializers
from .models import Admin

class AdminSerializer(serializers.ModelSerializer):
    class Meta:
        model = Admin
        fields = ['id', 'matricule', 'email', 'numéro_téléphone']
