from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from .models import Admin
from .serializers import AdminSerializer

class AdminViewSet(viewsets.ModelViewSet):
    queryset = Admin.objects.all()
    serializer_class = AdminSerializer

    @action(detail=False, methods=['post'], url_path='check-matricule')
    def check_matricule(self, request):
        matricule = request.data.get('matricule')

        if not matricule:
            return Response({'error': 'Matricule non fourni'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            admin = Admin.objects.get(matricule=matricule)
            return Response({'exists': True, 'nom': admin.nom}, status=status.HTTP_200_OK)
        except Admin.DoesNotExist:
            return Response({'exists': False}, status=status.HTTP_404_NOT_FOUND)
    @action(detail=False, methods=['get'], url_path='count')
    def get(self, request):
        # Compte le nombre d'administrateurs dans la table Admin
        admin_count = Admin.objects.count()
        return Response({"admin_count": admin_count}, status=status.HTTP_200_OK)