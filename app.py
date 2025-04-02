from flask import Flask, render_template
from modules.image import image_route
from modules.audio import audio_route
from modules.encryption import encryption_route
from modules.pdf import pdf_route
from modules.stéganalysis import stéganalysis_route 

app = Flask(__name__)

# Page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Importation des autres routes
app.register_blueprint(image_route, url_prefix='/steganography/image')
app.register_blueprint(audio_route, url_prefix='/steganography/audio')
app.register_blueprint(encryption_route, url_prefix='/encryption')
app.register_blueprint(pdf_route, url_prefix='/steganography/pdf')
app.register_blueprint(stéganalysis_route, url_prefix='/stegananalysis')  

if __name__ == '__main__':
    app.run(debug=True)
