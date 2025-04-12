import os
import io
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from typing import Dict, List

class PDFSteganalysis:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = self._load_pdf()
        self.images = []
        # Try to import CNN model if available
        self.cnn_model = self._try_load_cnn_model()

    def _try_load_cnn_model(self):
        try:
            # Try to import the model from the project structure
            import sys
            import os
            # Add the parent directory to path
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.stego_resnet import StegoResNet
            
            model = StegoResNet()
            # Assuming a pre-trained model is available
            # model.load_state_dict(torch.load("path/to/model.pth"))
            return model.eval()
        except Exception as e:
            print(f"[!] CNN model loading failed: {e}")
            return None

    def analyze(self) -> Dict:
        try:
            metadata_result = self._analyze_metadata()
            hidden_text_result = self._analyze_hidden_content()
            
            # Create basic results structure without CNN analysis
            results = {
                "metadata": {
                    "is_suspicious": metadata_result["is_stego"],
                    "verdict": metadata_result["verdict"],
                    "confidence": metadata_result["confidence"],
                    "details": metadata_result["details"]
                },
                "hidden_text": {
                    "count": hidden_text_result["details"]["count"],
                    "verdict": hidden_text_result["verdict"],
                    "confidence": hidden_text_result["confidence"],
                    "items": hidden_text_result["details"]["hidden_text"]
                }
            }
            
            # Only add image analysis if CNN model exists
            if self.cnn_model and self.images:
                image_analysis_result = self._analyze_images()
                results["image_analysis"] = {
                    "is_suspicious": image_analysis_result["is_stego"],
                    "verdict": image_analysis_result["verdict"],
                    "confidence": image_analysis_result["confidence"],
                    "details": image_analysis_result["details"]
                }
            else:
                # Add placeholder for image analysis when CNN is not available
                results["image_analysis"] = {
                    "is_suspicious": False,
                    "verdict": "CNN model not available for image analysis",
                    "confidence": 0.0,
                    "details": {"image_count": 0}
                }

            return results

        except Exception as e:
            return {"error": f"PDF analysis failed: {str(e)}"}

    # ──────────────── Analysis Steps ──────────────── #

    def _analyze_metadata(self) -> Dict:
        meta = self.doc.metadata or {}
        anomalies = []

        if meta.get("producer", "").lower() == "ghostscript":
            anomalies.append("Ghostscript producer may indicate manipulated PDF")
        if meta.get("creator", "").lower() != meta.get("producer", "").lower():
            anomalies.append("Creator/Producer mismatch")

        return {
            "verdict": "Suspicious metadata" if anomalies else "Metadata looks clean",
            "is_stego": bool(anomalies),
            "confidence": 0.8 if anomalies else 0.2,
            "details": {
                "metadata": meta,
                "anomalies": anomalies
            }
        }

    def _analyze_hidden_content(self) -> Dict:
        hidden_items = []
        object_stats = {"pages": len(self.doc), "objects": 0}
        
        for page in self.doc:
            blocks = page.get_text("dict").get("blocks", [])
            object_stats["objects"] += len(blocks)
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            color = span.get("color", 0)
                            size = span.get("size", 0)
                            if text and (color == 0xFFFFFF or size < 1.0):
                                hidden_items.append({
                                    "text": text,
                                    "page": page.number + 1,
                                    "color": color,
                                    "size": size
                                })
            
            # Extract images only if CNN model is available
            if self.cnn_model:
                self._extract_images_from_page(page)

        suspicious = len(hidden_items) > 0

        return {
            "verdict": "Hidden text detected" if suspicious else "No hidden text found",
            "is_stego": suspicious,
            "confidence": 0.9 if suspicious else 0.2,
            "details": {
                "count": len(hidden_items),
                "object_stats": object_stats,
                "hidden_text": hidden_items[:10]  # Limit to first 10
            }
        }

    def _extract_images_from_page(self, page):
        """Extract images from a PDF page for CNN analysis"""
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = self.doc.extract_image(xref)
            
            if base_image:
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    self.images.append({
                        "image": image,
                        "page": page.number + 1,
                        "index": img_index
                    })
                except Exception as e:
                    print(f"[!] Failed to load image: {e}")

    def _analyze_images(self) -> Dict:
        """Analyze extracted images using CNN model"""
        if not self.cnn_model or not self.images:
            return {
                "verdict": "No images to analyze",
                "is_stego": False,
                "confidence": 0.0,
                "details": {
                    "image_count": len(self.images)
                }
            }

        suspicious_images = []
        confidence_scores = []

        for img_data in self.images:
            img = img_data["image"]
            
            # Preprocess image for CNN input
            try:
                # Resize to expected input size (e.g., 224x224 for ResNet)
                img = img.convert("RGB").resize((224, 224))
                img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.cnn_model(img_tensor)
                    # Assuming binary classification (stego vs clean)
                    probs = torch.softmax(outputs, dim=1)
                    stego_prob = probs[0][1].item()  # Probability of being stego
                    
                    confidence_scores.append(stego_prob)
                    if stego_prob > 0.6:  # Threshold for considering suspicious
                        suspicious_images.append({
                            "page": img_data["page"],
                            "index": img_data["index"],
                            "confidence": round(stego_prob, 3)
                        })
            except Exception as e:
                print(f"[!] Image analysis failed: {e}")
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        is_suspicious = len(suspicious_images) > 0
        
        return {
            "verdict": f"{len(suspicious_images)} suspicious images detected" if is_suspicious else "No suspicious images found",
            "is_stego": is_suspicious,
            "confidence": round(avg_confidence, 3),
            "details": {
                "image_count": len(self.images),
                "suspicious_count": len(suspicious_images),
                "suspicious_images": suspicious_images[:10]  # Limit to first 10
            }
        }

    # ──────────────── Helpers ──────────────── #

    def _load_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        return fitz.open(self.pdf_path)
