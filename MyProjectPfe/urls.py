"""
URL configuration for MyProjectPfe project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from reconnaissance import views  # Remplace par ton fichier views

# Créer un routeur pour enregistrer les vues API
router = routers.DefaultRouter()
router.register(r'admins', views.AdminViewSet)  # Assure-toi que 'AdminViewSet' est bien défini dans 'views.py'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
]
