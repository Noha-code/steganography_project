import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50

class StegoResNet:
    """Modèle ResNet adapté pour la détection de stéganographie"""
    
    @staticmethod
    def build_model(input_shape=(224, 224, 3), weights='imagenet'):
        """
        Construit un modèle ResNet50 adapté à la stéganalyse
        
        Args:
            input_shape: Forme de l'entrée (hauteur, largeur, canaux)
            weights: Poids préentraînés à utiliser ('imagenet' ou None)
            
        Returns:
            Un modèle Keras compilé
        """
        # Base de ResNet50 sans les couches entièrement connectées
        base_model = ResNet50(
            weights=weights,
            include_top=False,
            input_shape=input_shape
        )
        
        # Pour la stéganalyse, nous voulons conserver les couches bas niveau
        # car elles détectent mieux les subtils changements de pixels
        for layer in base_model.layers[:50]:  # Geler les 50 premières couches
            layer.trainable = False
            
        # Ajouter nos propres couches pour la stéganalyse
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Ajout d'une couche de prétraitement au départ
        # Cette couche de prétraitement est cruciale pour la stéganalyse
        inputs = Input(shape=input_shape)
        preprocess = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        preprocess = layers.BatchNormalization()(preprocess)
        
        # Fusionner le prétraitement avec le modèle ResNet
        x_preprocess = base_model(preprocess)
        x_preprocess = layers.GlobalAveragePooling2D()(x_preprocess)
        
        # Combiner les caractéristiques
        x = layers.Concatenate()([x, x_preprocess])
        
        # Couches denses pour la classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Binaire: stégo ou non
        
        # Construire le modèle complet
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model

    @staticmethod
    def prepare_image(img_array, target_size=(224, 224)):
        """
        Prépare une image pour l'inférence
        
        Args:
            img_array: Image sous forme de tableau numpy
            target_size: Taille cible (hauteur, largeur)
            
        Returns:
            Image préparée pour le modèle
        """
        # Convertir en RGB si nécessaire
        if len(img_array.shape) == 2:  # Image en niveaux de gris
            img_array = tf.image.grayscale_to_rgb(tf.expand_dims(img_array, axis=-1))
        elif img_array.shape[-1] == 1:  # Image en niveaux de gris avec canal
            img_array = tf.image.grayscale_to_rgb(img_array)
        
        # Redimensionner
        img_array = tf.image.resize(img_array, target_size)
        
        # Normaliser
        img_array = img_array / 255.0
        
        # Ajouter la dimension du batch
        img_array = tf.expand_dims(img_array, axis=0)
        
        return img_array
