'''
main.py

Ce script permet de télécharger le jeu de données depuis sa source publique puis de créer les jeux de données depuis les métadonnées des images.
'''

# Téléchargement depuis le dépôt des données de Kaggle (nécessite des condentials)
od.download("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")

# Importation et concaténation des métadonnées disponibles
COVID_metadata = pd.read_excel('~/COVID.metadata.xlsx')
Lung_Opacity_metadata = pd.read_excel('~/Lung_Opacity.metadata.xlsx')
Normal_metadata = pd.read_excel('~/Normal.metadata.xlsx')
Pneumonia_metadata = pd.read_excel('~/Viral Pneumonia.metadata.xlsx')
df_metadata = pd.concat([COVID_metadata, Lung_Opacity_metadata, Normal_metadata, Pneumonia_metadata], ignore_index = True)

# Création de la colonne 'LABEL' à partir des noms de fichier
df_metadata['LABEL'] = df_metadata['FILE NAME'].str.split('-').str[0]

# Récupération des métadonnées des images
# Chemin vers le dossier contenant les catégories
root_folder = ''

# Initialiser une liste pour stocker les métadonnées
data = []

# Parcourir chaque catégorie de dossier
for category in ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']:
    images_folder = os.path.join(root_folder, category, 'images')
    if os.path.isdir(images_folder):
        for image_name in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_name)
            if os.path.isfile(image_path):
                # Ouvrir l'image pour obtenir des informations supplémentaires
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolution = f"{width}x{height}"
                    channels = img.mode
                    # Convertir le mode de l'image en nombre de canaux
                    # Généralement, 'RGB' -> 3, 'L' -> 1 (niveaux de gris), etc.
                    channels_count = len(channels)
                    # Ajouter les métadonnées à la liste
                    data.append({
                        'FILE NAME': image_name,
                        'FORMAT' : "",
                        'SIZE': resolution,
                        'LABEL': category,
                        'CHANNELS': channels_count,
                        'PATH': image_path})

# Créer un DataFrame à partir des métadonnées collectées et mise en forme des variables
df_images = pd.DataFrame(data)
df_images['FORMAT'] = df_images['FILE NAME'].str.split('.').str[1]
df_images['FILE NAME'] = df_images['FILE NAME'].str.split('.').str[0]
df_images['FORMAT'] = df_images['FORMAT'].str.upper()

# Définition d'une fonction pour extraire le nom de domaine des URL à disposition
def source_extract(url):
    pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
    match = pattern.search(url)
    if match:
        return match.group(1)
    else:
        return None

# Création de la colonne SOURCE
df_images = pd.merge(df_images, df_metadata[['FILE NAME', 'URL']], on = 'FILE NAME', how = 'left')
df_images['SOURCE'] = df_images['URL'].astype(str).apply(source_extract)