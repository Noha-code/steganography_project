import cv2
import numpy as np
from scipy.stats import chisquare, entropy
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import io
import os
import sys

# Import the StegoResNet model
sys.path.append("../../")  # Adjust path if necessary
try:
    from stego_resnet import StegoResNet
except ImportError:
    # Fallback if the import fails
    class StegoResNet:
        @staticmethod
        def build_model(input_shape=(224, 224, 3), weights='imagenet'):
            return None
            
        @staticmethod
        def prepare_image(img_array, target_size=(224, 224)):
            # Convertir en RGB si nécessaire
            if len(img_array.shape) == 2:  # Image en niveaux de gris
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            # Redimensionner
            img_array = cv2.resize(img_array, target_size)
            
            # Normaliser
            img_array = img_array / 255.0
            
            # Ajouter la dimension du batch
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array

class ImageSteganalysis:
    def __init__(self, cnn_model_path=None):
        """
        Initialize the steganalysis tool
        
        Args:
            cnn_model_path: Path to a pre-trained CNN model
        """
        self.cnn_model = self._load_cnn_model(cnn_model_path) if cnn_model_path else None

    def analyze(self, file_storage):
        """
        Analyze an image for steganography
        
        Args:
            file_storage: Image file object
            
        Returns:
            dict: Analysis results
        """
        try:
            image = self._load_image_from_filestorage(file_storage)
        except Exception as e:
            return {"error": f"Unable to read uploaded image: {str(e)}"}

        if image is None:
            return {"error": "Invalid or unsupported image file."}

        return self._run_analysis(image)

    def analyze_image_path(self, path):
        """
        Analyze an image at the given path
        
        Args:
            path: Path to the image file
            
        Returns:
            dict: Analysis results
        """
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}
            
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": f"Could not load image at: {path}"}
            
        return self._run_analysis(img)

    def _run_analysis(self, img):
        """
        Run all analysis methods on the image
        
        Args:
            img: Image as numpy array
            
        Returns:
            dict: Combined analysis results
        """
        with ThreadPoolExecutor() as executor:
            futures = {
                "LSB": executor.submit(self._analyze_lsb, img),
                "SPA": executor.submit(self._analyze_spa, img),
                "ChiSquare": executor.submit(self._analyze_chi_square, img),
            }
            if self.cnn_model:
                futures["CNN"] = executor.submit(self._analyze_cnn, img)

        # Récupérer tous les résultats
        results = {method: job.result() for method, job in futures.items()}
        
        # Calculer le résumé des résultats
        summary = self._summarize_results(results)
        
        return {"summary": summary, "results": results}

    # ──────────────── Méthodes d'analyse de stéganographie ──────────────── #

    def _analyze_lsb(self, img):
        """
        Analyse les bits de poids faible (LSB) pour détecter la stéganographie
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            dict: Résultats de l'analyse LSB
        """
        # Extraire le plan LSB
        lsb_plane = img & 1
        
        # Compter les occurrences de 0 et 1
        counts = np.bincount(lsb_plane.flatten(), minlength=2)
        total = img.size
        
        # Calculer le ratio entre 0 et 1 (proche de 0.5 pour une image stéganographiée)
        ratio = abs(counts[0] - counts[1]) / total
        
        # Calculer l'entropie du plan LSB
        lsb_entropy = entropy(counts / total) if counts[1] > 0 and counts[0] > 0 else 0
        
        # Déterminer si l'image contient de la stéganographie
        is_stego = ratio < 0.05 and lsb_entropy > 0.9
        
        return {
            "verdict": "Suspicious LSB pattern" if is_stego else "Normal LSB",
            "is_stego": is_stego,
            "confidence": round(min(1.0, max(0, 1 - (ratio / 0.1))), 3),
            "details": {
                "lsb_ratio": round(ratio, 4),
                "entropy": round(lsb_entropy, 4),
                "zeros": int(counts[0]),
                "ones": int(counts[1])
            }
        }

    def _analyze_spa(self, img):
        """
        Analyse par paires de pixels (Sample Pairs Analysis)
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            dict: Résultats de l'analyse SPA
        """
        # Calculer les différences entre pixels adjacents
        diff = np.abs(np.diff(img.astype(np.float32), axis=1))
        
        # Calculer le score SPA (moyenne des différences)
        spa_score = float(np.mean(diff))
        
        # Des différences trop faibles indiquent une possible stéganographie
        is_stego = spa_score < 2.0
        
        # Calculer la confiance (plus le score est bas, plus la confiance est élevée)
        # Limiter entre 0 et 1
        confidence = 1.0 - min(1.0, max(0, spa_score / 10.0))
        
        return {
            "verdict": "Abnormal pixel pairs" if is_stego else "Normal pixel distribution",
            "is_stego": is_stego,
            "confidence": round(confidence, 3),
            "details": {
                "spa_score": round(spa_score, 3),
                "mean_diff": round(spa_score, 3),
                "max_diff": float(np.max(diff)),
                "min_diff": float(np.min(diff))
            }
        }

    def _analyze_chi_square(self, img):
        """
        Test du chi-carré sur la distribution des bits de poids faible
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            dict: Résultats du test chi-carré
        """
        # Extraire le plan LSB
        lsb = img & 1
        
        # Compter les occurrences de 0 et 1
        counts = np.bincount(lsb.flatten(), minlength=2)
        
        # Distribution attendue pour un LSB non modifié
        expected = np.array([img.size / 2, img.size / 2])
        
        # Calculer la statistique du chi-carré
        try:
            chi2, p_value = chisquare(counts, f_exp=expected)
        except Exception:
            # En cas d'erreur, utiliser des valeurs par défaut
            chi2, p_value = 0, 1.0
        
        # Une p-value faible indique une distribution non aléatoire (stéganographie)
        is_stego = p_value < 0.05
        
        return {
            "verdict": "Non-random LSB distribution" if is_stego else "Random LSB pattern",
            "is_stego": is_stego,
            "confidence": round(max(0, min(1, 1 - p_value)), 3),
            "details": {
                "chi2_stat": float(round(chi2, 4)),
                "p_value": float(round(p_value, 4)),
                "lsb_zeros": int(counts[0]),
                "lsb_ones": int(counts[1])
            }
        }

    def _analyze_cnn(self, img):
        """
        Analyse avec un modèle de CNN pour détecter la stéganographie
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            dict: Résultats de l'analyse CNN
        """
        if self.cnn_model is None:
            return {
                "verdict": "CNN model not available",
                "is_stego": False,
                "confidence": 0.0,
                "details": {"error": "CNN model not loaded"}
            }
            
        try:
            # Utiliser la fonction de préparation d'image du modèle StegoResNet
            # Cette fonction convertit en RGB si nécessaire et normalise l'image
            prepared_img = StegoResNet.prepare_image(img, target_size=(224, 224))
            
            # Prédire
            prediction = self.cnn_model.predict(prepared_img, verbose=0)
            confidence = float(prediction[0][0])
            
            # Déterminer si l'image contient de la stéganographie
            is_stego = confidence > 0.5
            
            return {
                "verdict": "AI detected hidden data" if is_stego else "No AI anomalies",
                "is_stego": is_stego,
                "confidence": round(confidence, 3),
                "details": {"raw_prediction": float(round(confidence, 4))}
            }
        except Exception as e:
            return {
                "verdict": "CNN analysis failed",
                "is_stego": False,
                "confidence": 0.0,
                "details": {"error": str(e)}
            }

    # ──────────────── Fonctions auxiliaires ──────────────── #

    def _load_image_from_filestorage(self, file_storage):
        """
        Charge une image à partir d'un objet FileStorage ou d'un chemin de fichier
        
        Args:
            file_storage: Objet FileStorage ou chemin de fichier
            
        Returns:
            numpy.ndarray: Image en niveaux de gris
        """
        # Si c'est un chemin de fichier
        if isinstance(file_storage, str):
            if not os.path.exists(file_storage):
                raise FileNotFoundError(f"File not found: {file_storage}")
            return cv2.imread(file_storage, cv2.IMREAD_GRAYSCALE)
        
        # Si c'est un objet FileStorage
        try:
            in_memory_file = io.BytesIO()
            file_storage.save(in_memory_file)
            file_bytes = np.frombuffer(in_memory_file.getvalue(), np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def _load_cnn_model(self, model_path):
        """
        Charge un modèle CNN ou en construit un nouveau si nécessaire
        
        Args:
            model_path: Chemin vers un modèle pré-entraîné
            
        Returns:
            Un modèle TensorFlow
        """
        # Si nous avons un chemin de modèle valide, essayer de le charger
        if model_path and os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"[+] CNN model loaded successfully from {model_path}")
                return model
            except Exception as e:
                print(f"[!] Error loading CNN model: {e}")
                print("[*] Attempting to build a new StegoResNet model...")
        else:
            print(f"[!] CNN model path not found: {model_path}")
            print("[*] Building a new StegoResNet model...")
            
        # Si le chargement échoue ou si aucun chemin n'est fourni, essayer de construire un nouveau modèle
        try:
            # Créer un nouveau modèle StegoResNet
            model = StegoResNet.build_model(input_shape=(224, 224, 3), weights='imagenet')
            print("[+] New StegoResNet model built successfully")
            return model
        except Exception as e:
            print(f"[!] Error building StegoResNet model: {e}")
            return None

    def _summarize_results(self, results):
        """
        Résume les résultats de toutes les méthodes d'analyse
        
        Args:
            results: Dictionnaire des résultats par méthode
            
        Returns:
            dict: Résumé des résultats
        """
        # Compter les méthodes qui ont détecté de la stéganographie
        suspicious_methods = [name for name, data in results.items() if data.get("is_stego", False)]
        suspicious_count = len(suspicious_methods)
        
        # Calculer la confiance moyenne des méthodes qui ont détecté de la stéganographie
        if suspicious_count > 0:
            suspicious_confidences = [results[name]["confidence"] for name in suspicious_methods]
            average_confidence = sum(suspicious_confidences) / len(suspicious_confidences)
        else:
            average_confidence = 0.0
        
        # Calculer la confiance globale (basée sur toutes les méthodes)
        total_methods = len(results)
        overall_confidence = average_confidence * (suspicious_count / total_methods) if total_methods > 0 else 0
        
        return {
            "verdict": "Steganography DETECTED" if suspicious_count > 0 else "No steganography found",
            "confidence": round(overall_confidence, 3),
            "suspicious_methods": suspicious_methods,
            "total_methods": total_methods
        }
