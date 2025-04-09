import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Importer notre modèle défini précédemment
import sys
sys.path.append("../../")  # Ajuster si nécessaire
from modules.steganalysis.models.stego_resnet import StegoResNet

def create_synthetic_dataset(num_samples=1000, img_size=(224, 224)):
    """
    Crée un ensemble de données synthétiques pour l'entraînement initial
    
    Args:
        num_samples: Nombre d'images à générer
        img_size: Taille des images
        
    Returns:
        X_train, y_train: Images et étiquettes
    """
    X = []
    y = []
    
    for i in range(num_samples):
        # Créer une image aléatoire (normal)
        img = np.random.randint(0, 256, (img_size[0], img_size[1], 3), dtype=np.uint8)
        X.append(img)
        y.append(0)  # Étiquette: pas de stéganographie
        
        # Créer une version avec stéganographie (LSB)
        stego_img = img.copy()
        # Simuler une stéganographie LSB en modifiant les bits de poids faible
        # Dans un cas réel, on utiliserait un vrai algorithme de stéganographie
        mask = np.random.randint(0, 2, (img_size[0], img_size[1], 3), dtype=np.uint8)
        stego_img = stego_img & 0xFE | mask  # Remplacer les LSB par notre masque
        
        X.append(stego_img)
        y.append(1)  # Étiquette: contient de la stéganographie
    
    # Convertir en tableaux numpy
    X = np.array(X, dtype=np.float32) / 255.0  # Normaliser
    y = np.array(y, dtype=np.float32)
    
    return X, y

def load_real_dataset(data_dir):
    """
    Charge un ensemble de données réel pour l'entraînement
    
    Args:
        data_dir: Répertoire contenant les images
        
    Returns:
        X_train, y_train: Images et étiquettes
    """
    # Ici, nous supposons que data_dir contient deux sous-répertoires:
    # - 'normal': images sans stéganographie
    # - 'stego': images avec stéganographie
    
    X = []
    y = []
    
    # Charger les images normales
    normal_dir = os.path.join(data_dir, 'normal')
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            try:
                img_path = os.path.join(normal_dir, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                X.append(img_array)
                y.append(0)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Charger les images stéganographiées
    stego_dir = os.path.join(data_dir, 'stego')
    if os.path.exists(stego_dir):
        for filename in os.listdir(stego_dir):
            try:
                img_path = os.path.join(stego_dir, filename)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                X.append(img_array)
                y.append(1)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Convertir en tableaux numpy
    X = np.array(X, dtype=np.float32) / 255.0  # Normaliser
    y = np.array(y, dtype=np.float32)
    
    return X, y

def train_model(args):
    """
    Entraîne le modèle de stéganalyse
    
    Args:
        args: Arguments de ligne de commande
    """
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Charger ou créer les données
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"Loading real dataset from {args.data_dir}")
        X, y = load_real_dataset(args.data_dir)
    else:
        print(f"Creating synthetic dataset with {args.num_samples} samples")
        X, y = create_synthetic_dataset(args.num_samples)
    
    # Diviser en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Créer le modèle
    model = StegoResNet.build_model(input_shape=(224, 224, 3), weights='imagenet')
    
    # Configurer les callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'stego_resnet_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Augmentation de données pour l'entraînement
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )
    
    # Entraîner le modèle
    print("Training model...")
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Sauvegarder le modèle final
    model.save(os.path.join(args.output_dir, 'stego_resnet_final.h5'))
    print(f"Model saved to {os.path.join(args.output_dir, 'stego_resnet_final.h5')}")
    
    # Évaluer le modèle
    results = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation accuracy: {results[1]*100:.2f}%")
    print(f"Validation AUC: {results[2]*100:.2f}%")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a steganalysis model')
    parser.add_argument('--data_dir', type=str, default='', help='Directory containing real dataset')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for trained models')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of synthetic samples if no real data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    
    args = parser.parse_args()
    train_model(args)