# serializers.py
#Les sérialiseurs dans Django REST Framework transforment les objets du modèle en formats lisibles pour les API (comme JSON).
from rest_framework import serializers
from .models import Admin
from .models import Agence
class AdminSerializer(serializers.ModelSerializer):
    class Meta:
        model = Admin
        fields = ['id', 'matricule', 'email', 'numéro_téléphone']

from .models import Agence

class AgenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Agence
        fields = ['nom', 'adresse', 'latitude', 'longitude', 'telephone', 'horaire_debut', 'horaire_fin', 'jours_ouverture', 'statut']
# Vue pour l'API
