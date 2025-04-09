from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from modules.steganalysis.img_steganalysis import ImageSteganalysis
from modules.steganalysis.pdf_steganalysis import PDFSteganalysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def handle_image_analysis(filepath):
    """
    Analyse une image pour détecter la stéganographie
    Adapte les résultats de ImageSteganalysis au format attendu par les templates
    """
    analyzer = ImageSteganalysis()  
    analysis_results = analyzer.analyze_image_path(filepath)  # Utiliser analyze_image_path au lieu de analyze
    
    if "error" in analysis_results:
        return {"file_type": 'image', "error": analysis_results["error"]}
    
    # Extraction des résultats à partir de la structure correcte
    results = analysis_results["results"]
    
    methods = {
        'LSB': {
            'verdict': results['LSB']['verdict'],
            'confidence': int(results['LSB']['confidence'] * 100),  # Convertir en pourcentage
            'details': results['LSB']['details']
        },
        'SPA': {
            'verdict': results['SPA']['verdict'],
            'confidence': int(results['SPA']['confidence'] * 100),
            'details': results['SPA']['details']
        },
        'ChiSquare': {
            'verdict': results['ChiSquare']['verdict'],
            'confidence': int(results['ChiSquare']['confidence'] * 100),
            'details': results['ChiSquare']['details']
        }
    }
    
    # Ajouter CNN seulement s'il est présent dans les résultats
    if 'CNN' in results:
        methods['CNN'] = {
            'verdict': results['CNN']['verdict'],
            'confidence': int(results['CNN']['confidence'] * 100),
            'details': results['CNN']['details']
        }
    
    combined_confidence = calculate_combined_confidence(methods)
    return {
        'file_type': 'image',
        'combined_confidence': combined_confidence,
        'methods': methods
    }

def handle_pdf_analysis(filepath):
    """
    Analyse un PDF pour détecter la stéganographie
    Adapte les résultats de PDFSteganalysis au format attendu par les templates
    """
    analyzer = PDFSteganalysis(filepath)
    analysis_results = analyzer.analyze()
    
    if "error" in analysis_results:
        return {"file_type": 'pdf', "error": analysis_results["error"]}
    
    methods = {
        'Metadata': {
            'verdict': analysis_results['metadata']['verdict'],
            'confidence': int(analysis_results['metadata']['confidence'] * 100),
            'details': analysis_results['metadata']['details']
        },
        'Hidden Text': {
            'verdict': f"{analysis_results['hidden_text']['count']} hidden text elements found",
            'confidence': min(100, analysis_results['hidden_text']['count'] * 20),
            'details': analysis_results['hidden_text']
        },
        'Text Analysis': {
            'verdict': analysis_results['text_analysis']['verdict'],
            'confidence': int(analysis_results['text_analysis']['confidence'] * 100),
            'details': analysis_results['text_analysis']['details']
        }
    }
    
    combined_confidence = calculate_pdf_confidence(methods)
    return {
        'file_type': 'pdf',
        'combined_confidence': combined_confidence,
        'methods': methods
    }
    
    combined_confidence = calculate_pdf_confidence(methods)
    return {
        'file_type': 'pdf',
        'combined_confidence': combined_confidence,
        'methods': methods
    }

@app.route('/steganalysis')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])
def steganalysis():
    if request.method == 'POST':
        # Vérifier si un fichier a été envoyé
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        # Vérifier si un fichier a été sélectionné
        if file.filename == '':
            return redirect(request.url)
            
        # Vérifier si le fichier est autorisé
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # S'assurer que le dossier existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Enregistrer le fichier
            file.save(filepath)

            # Analyser le fichier selon son type
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                results = handle_image_analysis(filepath)
            elif filename.lower().endswith('.pdf'):
                results = handle_pdf_analysis(filepath)
            else:
                return "Unsupported file type", 400

            # Afficher les résultats
            if "error" in results:
                return render_template('error.html', error=results["error"])
                
            return render_template('results.html', filename=filename, results=results)

    return render_template('analyze.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/sniffing')
def sniffing():
    return render_template('sniff_traffic.html')

def calculate_combined_confidence(methods):
    """
    Calcule la confiance combinée pour les analyses d'images
    """
    weights = {
        'LSB': 0.3,
        'SPA': 0.25,
        'ChiSquare': 0.2,
        'CNN': 0.25
    }
    
    total_confidence = 0
    total_weight = 0
    
    for method, data in methods.items():
        if method in weights:
            total_confidence += data['confidence'] * weights[method]
            total_weight += weights[method]
    
    # Ajuster si certaines méthodes sont absentes
    if total_weight > 0:
        return int(total_confidence / total_weight)
    else:
        return 0

def calculate_pdf_confidence(methods):
    """
    Calcule la confiance combinée pour les analyses de PDF
    """
    return int(
        methods['Metadata']['confidence'] * 0.3 +
        methods['Hidden Text']['confidence'] * 0.4 +
        methods['Text Analysis']['confidence'] * 0.3
    )

if __name__ == '__main__':
    # Créer le dossier d'upload s'il n'existe pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
