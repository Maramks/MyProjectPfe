from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from .models import Admin
from .serializers import AdminSerializer
from .models import Agence
from .serializers import AgenceSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Agence

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
    def get_admin_count(self, request):
        # Compte le nombre d'administrateurs dans la table Admin
        admin_count = Admin.objects.count()
        return Response({"admin_count": admin_count}, status=status.HTTP_200_OK)


class AgenceViewSet(viewsets.ModelViewSet):
    queryset = Agence.objects.all()
    serializer_class = AgenceSerializer 
    def list(self, request):
        agences = Agence.objects.all()

        if not agences.exists():
            return Response(
                {"message": "Aucune agence trouvée."},
                status=status.HTTP_404_NOT_FOUND
            )
        # Mettre à jour le statut avant de retourner les agences
        for agence in agences:
         agence.est_ouvert()

        serializer = AgenceSerializer(agences, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request):
        serializer = AgenceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(
            {"error": "Données invalides", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )

    def update(self, request, pk=None):
        try:
            agence = Agence.objects.get(pk=pk)
        except Agence.DoesNotExist:
            return Response(
                {"message": "Agence non trouvée."},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = AgenceSerializer(agence, data=request.data, partial=True)  # Utilisation de partial=True pour les mises à jour partielles
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(
            {"error": "Données invalides", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )

    def destroy(self, request, pk=None):
        try:
            agence = Agence.objects.get(pk=pk)
            agence.delete()
            return Response(
                {"message": "Agence supprimée avec succès."},
                status=status.HTTP_204_NO_CONTENT
            )
        except Agence.DoesNotExist:
            return Response(
                {"message": "Agence non trouvée."},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"Une erreur est survenue : {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'], url_path='count')
    def count(self, request):
        count_agences = Agence.objects.count()
        return Response(
            {"count": count_agences},
            status=status.HTTP_200_OK
        )
# tester si l'image est floue ou non 
import cv2
import numpy as np
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

class ImageBlurDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # Vérifier si une image a été envoyée
        if 'image' not in request.FILES:
            return Response({"error": "Aucune image envoyée."}, status=status.HTTP_400_BAD_REQUEST)

        # Lire l'image sans la sauvegarder
        image_file = request.FILES['image'].read()
        np_array = np.frombuffer(image_file, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

        # Appliquer un flou de Gaussian pour réduire les bruits de l'image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Calculer la variance du Laplacien de l'image (c'est une mesure de la netteté)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        variance = laplacian.var()

        # Ajuster le seuil pour détecter même de faibles niveaux de flou
        blur_threshold = 50.0  # Baisser le seuil pour détecter les images légèrement floues
        is_blurry = variance < blur_threshold

        # Message conditionnel en fonction du flou
        if is_blurry:
            message = "Image floue"
        else:
            message = "Image claire"

        # Réponse JSON avec les détails
        return Response({
            "is_blurry": is_blurry,
            "blur_value": variance,
            "message": message  # Afficher si l'image est floue ou claire
        })
# reconnaissance image texte
import os
import cv2
import numpy as np
import pytesseract
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.conf import settings

# Définir le chemin de l'image navbar en utilisant BASE_DIR
NAVBAR_IMAGE_PATH = os.path.join(settings.BASE_DIR, "static", "maram.jpg")

# Vérifier si l'image du navbar existe
if not os.path.exists(NAVBAR_IMAGE_PATH):
    raise FileNotFoundError(f"Image de navbar non trouvée : {NAVBAR_IMAGE_PATH}")

# Fonction pour détecter le navbar avec ORB et FLANN
def detect_navbar_with_orb(image):
    try:
        # Chargement de l'image du navbar
        navbar_ref = cv2.imread(NAVBAR_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if navbar_ref is None:
            raise ValueError(f"Erreur lors du chargement de l'image de navbar depuis {NAVBAR_IMAGE_PATH}.")

        # Convertir l'image reçue en niveaux de gris
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Créer un détecteur ORB avec un nombre plus élevé de points clés
        orb = cv2.ORB_create(nfeatures=1500)  # Augmentation du nombre de points clés

        # Détection des points clés et des descripteurs
        kp1, des1 = orb.detectAndCompute(navbar_ref, None)
        kp2, des2 = orb.detectAndCompute(image_gray, None)

        # Vérifier s'il y a des descripteurs détectés
        if des1 is None or des2 is None:
            return False, None

        # Utiliser FLANN pour trouver des correspondances entre les descripteurs
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=100)  # Augmenter le nombre de checks pour une recherche plus exhaustive
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Appliquer la méthode de filtrage des bonnes correspondances
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Ratio de Lowe (0.7 est une bonne valeur)
                good_matches.append(m)

        # Si assez de bonnes correspondances sont trouvées
        if len(good_matches) > 15:  # Augmenter le seuil pour avoir plus de correspondances
            print(f"Navbar détecté avec {len(good_matches)} bonnes correspondances.")
            # Dessiner les correspondances sur l'image
            img_matches = cv2.drawMatches(navbar_ref, kp1, image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Sauvegarder l'image annotée pour débogage
            debug_image_path = os.path.join(settings.BASE_DIR, "static", "debug_navbar_detected_orb.jpg")
            cv2.imwrite(debug_image_path, img_matches)
            
            return True, debug_image_path
        else:
            print("Pas assez de bonnes correspondances pour détecter le navbar.")
            return False, None

    except Exception as e:
        print(f"Erreur dans la détection du navbar avec ORB: {e}")
        return False, None

# Fonction pour extraire le texte sans le navbar
def extract_text_from_other_regions(image):
    try:
        height, width = image.shape[:2]
        image_without_navbar = image[height // 3:, :]  # Découpe l'image sous le navbar

        # Appliquer Tesseract pour extraire le texte de la partie restante
        extracted_text = pytesseract.image_to_string(image_without_navbar)
        return extracted_text.strip()
    except Exception as e:
        print(f"Erreur dans l'extraction du texte: {e}")
        return ""

@api_view(['POST'])
def detect_cin(request):
    try:
        # Vérifier si l'image est bien envoyée
        if 'image' not in request.FILES:
            return JsonResponse({"status": "error", "message": "Aucune image envoyée"}, status=400)

        uploaded_image = request.FILES['image']
        image_array = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return JsonResponse({"status": "error", "message": "Impossible de lire l'image"}, status=400)

        print(f"Image reçue avec forme: {image.shape}")

        # Vérification de la présence du navbar
        is_navbar_detected, debug_image_path = detect_navbar_with_orb(image)

        if not is_navbar_detected:
            return JsonResponse({"status": "error", "is_cin": False, "message": "Navbar non détecté, ce n'est pas une CIN"})

        # Extraction du texte (ajustée pour ignorer la partie navbar)
        extracted_text = extract_text_from_other_regions(image)

        return JsonResponse({
            "status": "success",
            "is_cin": True,
            "extracted_text": extracted_text,
            "debug_image": debug_image_path  # Vous pouvez voir l'image avec la détection du navbar ici
        })

    except Exception as e:
        print(f"Erreur survenue dans l'API: {str(e)}")
        return JsonResponse({"status": "error", "message": f"Une erreur est survenue: {str(e)}"}, status=500)
