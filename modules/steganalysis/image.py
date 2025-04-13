import cv2
import numpy as np
from scipy.stats import chisquare, entropy
import torch
from concurrent.futures import ThreadPoolExecutor
import io
import os
import sys

# Import the StegoResNet model
sys.path.append("../../")  # Adjust path if necessary
try:
    from models.stego_resnet import StegoResNet
except ImportError:
    # Fallback if the import fails
    class StegoResNet:
        @staticmethod
        def build_model(input_shape=(224, 224, 3), weights='imagenet'):
            return None
            
        @staticmethod
        def prepare_image(img_array, target_size=(224, 224)):
            # Convert to RGB if necessary
            if len(img_array.shape) == 2:  # Grayscale image
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            # Resize
            img_array = cv2.resize(img_array, target_size)
            
            # Normalize
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array

class ImageSteganalysis:
    def __init__(self, cnn_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = self._load_cnn_model(cnn_model_path) if cnn_model_path else None

    def analyze(self, file_storage):
        try:
            image = self._load_image_from_filestorage(file_storage)
        except Exception as e:
            return {"error": f"Unable to read uploaded image: {str(e)}"}

        if image is None:
            return {"error": "Invalid or unsupported image file."}

        return self._run_analysis(image)

    def analyze_image_path(self, path):
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

        # Get all results
        results = {method: job.result() for method, job in futures.items()}
        
        # Calculate summary
        summary = self._summarize_results(results)
        
        return {"summary": summary, "results": results}

    # ──────────────── Steganography analysis methods ──────────────── #

    def _analyze_lsb(self, img):
        
        # Extract LSB plane
        lsb_plane = img & 1
        
        # Count occurrences of 0 and 1
        counts = np.bincount(lsb_plane.flatten(), minlength=2)
        total = img.size
        
        # Calculate ratio between 0 and 1 (close to 0.5 for a steganographed image)
        ratio = abs(counts[0] - counts[1]) / total
        
        # Calculate entropy of LSB plane
        lsb_entropy = entropy(counts / total) if counts[1] > 0 and counts[0] > 0 else 0
        
        # Determine if image contains steganography
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
        Sample Pairs Analysis
        
        Args:
            img: Grayscale image
            
        Returns:
            dict: SPA analysis results
        """
        # Calculate differences between adjacent pixels
        diff = np.abs(np.diff(img.astype(np.float32), axis=1))
        
        # Calculate SPA score (mean of differences)
        spa_score = float(np.mean(diff))
        
        # Too small differences indicate possible steganography
        is_stego = spa_score < 2.0
        
        # Calculate confidence (lower score means higher confidence)
        # Limit between 0 and 1
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
        Chi-square test on LSB distribution
        
        Args:
            img: Grayscale image
            
        Returns:
            dict: Chi-square test results
        """
        # Extract LSB plane
        lsb = img & 1
        
        # Count occurrences of 0 and 1
        counts = np.bincount(lsb.flatten(), minlength=2)
        
        # Expected distribution for unmodified LSB
        expected = np.array([img.size / 2, img.size / 2])
        
        # Calculate chi-square statistic
        try:
            chi2, p_value = chisquare(counts, f_exp=expected)
        except Exception:
            # In case of error, use default values
            chi2, p_value = 0, 1.0
        
        # A low p-value indicates non-random distribution (steganography)
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
        Analyze with CNN model to detect steganography
        
        Args:
            img: Grayscale image
            
        Returns:
            dict: CNN analysis results
        """
        if self.cnn_model is None:
            return {
                "verdict": "CNN model not available",
                "is_stego": False,
                "confidence": 0.0,
                "details": {"error": "CNN model not loaded"}
            }
            
        try:
            # Use StegoResNet's image preparation function
            # This converts to RGB if necessary and normalizes the image
            prepared_img = StegoResNet.prepare_image(img, target_size=(224, 224))
            
            # Move tensor to the appropriate device
            prepared_img = prepared_img.to(self.device)
            
            # Set model to evaluation mode
            self.cnn_model.eval()
            
            # Predict
            with torch.no_grad():
                prediction = self.cnn_model(prepared_img)
            confidence = float(prediction.cpu().numpy()[0][0])
            
            # Determine if image contains steganography
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

    # ──────────────── Helper functions ──────────────── #

    def _load_image_from_filestorage(self, file_storage):
        """
        Loads an image from a FileStorage object or file path
        
        Args:
            file_storage: FileStorage object or file path
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        # If it's a file path
        if isinstance(file_storage, str):
            if not os.path.exists(file_storage):
                raise FileNotFoundError(f"File not found: {file_storage}")
            return cv2.imread(file_storage, cv2.IMREAD_GRAYSCALE)
        
        # If it's a FileStorage object
        try:
            in_memory_file = io.BytesIO()
            file_storage.save(in_memory_file)
            file_bytes = np.frombuffer(in_memory_file.getvalue(), np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def _load_cnn_model(self, model_path):
        """
        Loads a CNN model or builds a new one if necessary
        
        Args:
            model_path: Path to a pre-trained model
            
        Returns:
            A PyTorch model
        """
        # If we have a valid model path, try to load it
        if model_path and os.path.exists(model_path):
            try:
                # Create model instance first
                model = StegoResNet.build_model(input_shape=(224, 224, 3), weights=None)
                # Load the saved state dictionary
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model = model.to(self.device)
                model.eval()  # Set to evaluation mode
                print(f"[+] CNN model loaded successfully from {model_path}")
                return model
            except Exception as e:
                print(f"[!] Error loading CNN model: {e}")
                print("[*] Attempting to build a new StegoResNet model...")
        else:
            print(f"[!] CNN model path not found: {model_path}")
            print("[*] Building a new StegoResNet model...")
            
        # If loading fails or no path is provided, try to build a new model
        try:
            # Create a new StegoResNet model
            model = StegoResNet.build_model(input_shape=(224, 224, 3), weights='imagenet')
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            print("[+] New StegoResNet model built successfully")
            return model
        except Exception as e:
            print(f"[!] Error building StegoResNet model: {e}")
            return None

    def _summarize_results(self, results):
        """
        Summarizes results from all analysis methods
        
        Args:
            results: Dictionary of results by method
            
        Returns:
            dict: Summary of results
        """
        # Count methods that detected steganography
        suspicious_methods = [name for name, data in results.items() if data.get("is_stego", False)]
        suspicious_count = len(suspicious_methods)
        
        # Calculate average confidence of methods that detected steganography
        if suspicious_count > 0:
            suspicious_confidences = [results[name]["confidence"] for name in suspicious_methods]
            average_confidence = sum(suspicious_confidences) / len(suspicious_confidences)
        else:
            average_confidence = 0.0
        
        # Calculate overall confidence (based on all methods)
        total_methods = len(results)
        overall_confidence = average_confidence * (suspicious_count / total_methods) if total_methods > 0 else 0
        
        return {
            "verdict": "Steganography DETECTED" if suspicious_count > 0 else "No steganography found",
            "confidence": round(overall_confidence, 3),
            "suspicious_methods": suspicious_methods,
            "total_methods": total_methods
        }
