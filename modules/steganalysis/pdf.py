
import os
import io
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PDFSteganalysis:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = self._load_pdf()
        self.text = ""
        self.nlp_model, self.tokenizer = self._load_nlp_model()

    def analyze(self) -> Dict:
        
        try:
            metadata_result = self._analyze_metadata()
            hidden_text_result = self._analyze_hidden_content()
            text_analysis_result = self._analyze_text()

            # Format adapté aux attentes de handle_pdf_analysis() dans app.py
            return {
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
                },
                "text_analysis": {
                    "is_suspicious": text_analysis_result["is_stego"],
                    "verdict": text_analysis_result["verdict"],
                    "confidence": text_analysis_result["confidence"],
                    "details": text_analysis_result["details"]
                }
            }

        except Exception as e:
            return {"error": f"Analyse du PDF échouée: {str(e)}"}

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
        full_text = ""

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
                            if text:
                                full_text += text + " "
                                if color == 0xFFFFFF or size < 1.0:
                                    hidden_items.append({
                                        "text": text,
                                        "page": page.number + 1,
                                        "color": color,
                                        "size": size
                                    })

        self.text = full_text.strip()
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

    def _analyze_text(self) -> Dict:
        if not self.nlp_model or not self.text:
            return {
                "verdict": "NLP model not available or no text extracted",
                "is_stego": False,
                "confidence": 0.0,
                "details": {}
            }

        inputs = self.tokenizer(self.text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item()
        is_suspicious = confidence < 0.7

        return {
            "verdict": "Text anomaly detected" if is_suspicious else "Text appears normal",
            "is_stego": is_suspicious,
            "confidence": round(1 - confidence if is_suspicious else confidence, 3),
            "details": {
                "text_length": len(self.text),
                "model_confidence": round(confidence, 3)
            }
        }

    # ──────────────── Helpers ──────────────── #

    def _load_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        return fitz.open(self.pdf_path)

    def _load_nlp_model(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            return model.eval(), tokenizer
        except Exception as e:
            print(f"[!] NLP model loading failed: {e}")
            return None, None
