from flask import Flask, render_template, request, send_file
import os
from PyPDF2 import PdfReader, PdfWriter
from werkzeug.utils import secure_filename
import io
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session security
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Fonction pour dériver la clé de chiffrement à partir du mot de passe ---
def derive_key(password: str):
    salt = b'\x00' * 16  # Utiliser un sel fixe ou unique par utilisateur pour la sécurité
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

# --- Fonction pour chiffrer le message ---
def encrypt_message(message: str, password: str):
    key = derive_key(password)
    iv = os.urandom(16)  # Initialisation vector (IV) pour AES
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Padding pour s'assurer que le message est un multiple de 16 bytes
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()

    encrypted_message = encryptor.update(padded_data) + encryptor.finalize()

    return base64.b64encode(iv + encrypted_message).decode('utf-8')

# --- Fonction pour déchiffrer le message ---
def decrypt_message(encrypted_message: str, password: str):
    data = base64.b64decode(encrypted_message)
    iv = data[:16]  # Le premier 16 bytes sont l'IV
    encrypted_message = data[16:]

    key = derive_key(password)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    padded_message = decryptor.update(encrypted_message) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    original_message = unpadder.update(padded_message) + unpadder.finalize()

    return original_message.decode()

# --- Fonction pour cacher le message dans les métadonnées ---
def hide_in_metadata(pdf_path, message, password):
    encrypted_message = encrypt_message(message, password)
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    metadata = reader.metadata or {}
    metadata.update({"/SteganoMessage": encrypted_message})
    writer.add_metadata(metadata)

    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output

# --- Fonction pour extraire et déchiffrer le message des métadonnées ---
def extract_from_metadata(pdf_path, password):
    reader = PdfReader(pdf_path)
    metadata = reader.metadata
    encrypted_message = metadata.get("/SteganoMessage", None)

    if encrypted_message:
        return decrypt_message(encrypted_message, password)
    return "[Aucun message trouvé]"

# --- Routes ---
@app.route('/')
def index():
    return render_template('steg-pdf/index.html')

@app.route('/hide_page')
def hide_page():
    return render_template('steg-pdf/hide_page.html')

@app.route('/extract_page')
def extract_page():
    return render_template('steg-pdf/extract_page.html')

@app.route('/hide', methods=['POST'])
def hide():
    pdf = request.files['pdf_file']
    message = request.form['message']
    password = request.form['password']
    filename = secure_filename(pdf.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(path)

    output = hide_in_metadata(path, message, password)
    return send_file(output, as_attachment=True, download_name="pdf_cache.pdf")

@app.route('/extract', methods=['POST'])
def extract():
    pdf = request.files['pdf_file']
    password = request.form['password']
    filename = secure_filename(pdf.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(path)

    message = extract_from_metadata(path, password)
    return f"<h1>Message extrait :</h1><p>{message}</p><a href='/'>⬅ Retour</a>"

if __name__ == '__main__':
    app.run(debug=True)
