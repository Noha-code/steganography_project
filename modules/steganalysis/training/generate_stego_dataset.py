import os
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def create_directory_structure(output_dir):
    """
    Crée la structure de répertoire pour le dataset
    
    Args:
        output_dir: Répertoire principal de sortie
    """
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'stego'), exist_ok=True)
    print(f"Created directory structure in {output_dir}")

def embed_lsb_message(image, message=None, embed_ratio=0.1):
    """
    Intègre un message dans l'image en utilisant la méthode LSB
    
    Args:
        image: Image sous forme de tableau numpy
        message: Message à intégrer (optionnel)
        embed_ratio: Pourcentage de pixels à modifier si pas de message
        
    Returns:
        Image avec message intégré
    """
    # Faire une copie de l'image
    stego_img = np.copy(image)
    
    height, width, channels = image.shape
    total_pixels = height * width * channels
    
    if message:
        # Convertir le message en binaire
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        # Ajouter un délimiteur de fin
        binary_message += '00000000'  # ASCII NULL
        
        if len(binary_message) > total_pixels:
            binary_message = binary_message[:total_pixels]
            print("Warning: Message truncated to fit image")
        
        # Intégrer le message binaire
        idx = 0
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    if idx < len(binary_message):
                        # Remplacer le LSB par le bit du message
                        stego_img[i, j, c] = (stego_img[i, j, c] & 0xFE) | int(binary_message[idx])
                        idx += 1
                    else:
                        break
    else:
        # Modifier un pourcentage aléatoire de LSB
        num_pixels_to_modify = int(total_pixels * embed_ratio)
        random_indices = np.random.choice(total_pixels, num_pixels_to_modify, replace=False)
        
        for idx in random_indices:
            i = idx // (width * channels)
            j = (idx % (width * channels)) // channels
            c = idx % channels
            
            # Inverser le LSB
            stego_img[i, j, c] = stego_img[i, j, c] ^ 1
    
    return stego_img

def create_stego_dataset_from_images(input_dir, output_dir, num_samples, embed_ratio=0.1):
    """
    Crée un dataset à partir d'images existantes
    
    Args:
        input_dir: Répertoire contenant les images source
        output_dir: Répertoire de sortie
        num_samples: Nombre d'échantillons à générer
        embed_ratio: Pourcentage de bits à modifier
    """
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    create_directory_structure(output_dir)
    
    # Générer les échantillons
    for i in tqdm(range(num_samples), desc="Generating dataset"):
        # Choisir une image source aléatoire
        img_file = np.random.choice(image_files)
        img_path = os.path.join(input_dir, img_file)
        
        try:
            # Charger l'image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Redimensionner si nécessaire
            if img.shape[0] > 300 or img.shape[1] > 300:
                img = cv2.resize(img, (224, 224))
            
            # Enregistrer la version normale
            normal_output = os.path.join(output_dir, 'normal', f"normal_{i:04d}.png")
            cv2.imwrite(normal_output, img)
            
            # Créer la version stéganographiée
            stego_img = embed_lsb_message(img, embed_ratio=embed_ratio)
            stego_output = os.path.join(output_dir, 'stego', f"stego_{i:04d}.png")
            cv2.imwrite(stego_output, stego_img)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print("Dataset generation complete")

def create_synthetic_dataset(output_dir, num_samples, img_size=(224, 224), embed_ratio=0.1):
    """
    Crée un dataset synthétique
    
    Args:
        output_dir: Répertoire de sortie
        num_samples: Nombre d'échantillons à générer
        img_size: Taille des images
        embed_ratio: Pourcentage de bits à modifier
    """
    create_directory_structure(output_dir)
    
    for i in tqdm(range(num_samples), desc="Generating synthetic dataset"):
        # Générer une image avec du bruit, des dégradés et des formes
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # Ajouter un dégradé de fond
        for y in range(img_size[0]):
            for x in range(img_size[1]):
                img[y, x, 0] = int(255 * (x / img_size[1]))  # R
                img[y, x, 1] = int(255 * (y / img_size[0]))  # G
                img[y, x, 2] = int(255 * ((x+y) / (img_size[0]+img_size[1])))  # B
        
        # Ajouter des formes aléatoires
        for _ in range(np.random.randint(3, 10)):
            color = tuple(np.random.randint(0, 256, 3).tolist())
            center = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
            radius = np.random.randint(10, 50)
            thickness = np.random.randint(-1, 5)  # -1 = rempli
            shape_type = np.random.randint(0, 3)
            
            if shape_type == 0:  # Cercle
                cv2.circle(img, center, radius, color, thickness)
            elif shape_type == 1:  # Rectangle
                pt1 = center
                pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
                cv2.rectangle(img, pt1, pt2, color, thickness)
            else:  # Ligne
                pt1 = center
                pt2 = (np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]))
                cv2.line(img, pt1, pt2, color, np.random.randint(1, 5))
        
        # Ajouter un léger bruit de grain
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Enregistrer la version normale
        normal_output = os.path.join(output_dir, 'normal', f"normal_{i:04d}.png")
        cv2.imwrite(normal_output, img)
        
        # Créer la version stéganographiée
        stego_img = embed_lsb_message(img, embed_ratio=embed_ratio)
        stego_output = os.path.join(output_dir, 'stego', f"stego_{i:04d}.png")
        cv2.imwrite(stego_output, stego_img)
    
    print("Synthetic dataset generation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate steganalysis dataset')
    parser.add_argument('--input_dir', type=str, default='', help='Directory containing source images (optional)')
    parser.add_argument('--output_dir', type=str, default='./stego_dataset', help='Output directory for dataset')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--embed_ratio', type=float, default=0.1, help='Ratio of pixels to modify for steganography')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic images instead of using real ones')
    
    args = parser.parse_args()
    
    if args.synthetic or not args.input_dir:
        create_synthetic_dataset(args.output_dir, args.num_samples, embed_ratio=args.embed_ratio)
    else:
        create_stego_dataset_from_images(args.input_dir, args.output_dir, args.num_samples, embed_ratio=args.embed_ratio)