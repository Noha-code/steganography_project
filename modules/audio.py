from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from pydub import AudioSegment
from Crypto.Cipher import AES
import base64
import os
import numpy as np
from scipy.io import wavfile

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Nécessaire pour utiliser flash()

# Dossier de téléchargement des fichiers audio
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fonction pour ajouter un padding au message
def pad_message(message):
    return message + (16 - len(message) % 16) * ' '

# Fonction de chiffrement AES
def encrypt_message(message, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(pad_message(message).encode())).decode()

# Fonction de déchiffrement AES
def decrypt_message(encrypted_message, key):
    cipher = AES.new(key, AES.MODE_ECB)
    try:
        return cipher.decrypt(base64.b64decode(encrypted_message)).decode().strip()
    except Exception:
        return None  # erreur de déchiffrement

# Fonction pour cacher le message dans un fichier audio WAV
def hide_message(audio_path, message, key, output_wav_path):
    encrypted_message = encrypt_message(message, key) + "###"  # Message chiffré
    message_bin = ''.join(format(ord(c), '08b') for c in encrypted_message)

    sample_rate, data = wavfile.read(audio_path)
    if data.dtype != np.int16:
        raise ValueError("Seuls les fichiers audio PCM 16 bits sont supportés.")

    if len(message_bin) > len(data):
        raise ValueError("Le fichier audio est trop petit pour contenir ce message.")

    data = data.astype(np.int16)
    for i in range(len(message_bin)):
        data[i] = (data[i] & ~1) | int(message_bin[i])

    wavfile.write(output_wav_path, sample_rate, data)

# Fonction pour extraire un message caché depuis un fichier audio WAV
def extract_message(audio_path, key):
    print(f"[Début extraction] depuis : {audio_path}")
    
    # Lecture du fichier audio WAV
    _, data = wavfile.read(audio_path)

    # Extraction des bits
    bits = ''.join(str(sample & 1) for sample in data)
    
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(byte, 2))
        chars.append(char)
        if ''.join(chars).endswith("###"):
            break

    # Récupérer le message caché
    secret_message = ''.join(chars)
    encrypted_message = secret_message.replace("###", "")
    
      # Déchiffrer le message extrait
    return decrypt_message(encrypted_message, key)

# Route pour la page d'accueil
@app.route('/audio')
def index():
    return render_template('steg-audio/index.html')

# Route pour la page d'encodage (cacher un message)
@app.route('/encode')
def encode_page():
    return render_template('steg-audio/encode.html')

# Route pour la page de décodage (extraire un message)
@app.route('/decode')
def decode_page():
    return render_template('steg-audio/decode.html')

# Route POST pour encoder (cacher un message dans l'audio)
@app.route('/encode', methods=['POST'])
def encode():
    try:
        file = request.files['audio']
        message = request.form['message']
        key = request.form['key'].ljust(16)[:16].encode()

        # Enregistrement du fichier audio original
        original_ext = os.path.splitext(file.filename)[1].lower()
        original_path = os.path.join(UPLOAD_FOLDER, file.filename)
        wav_path = os.path.join(UPLOAD_FOLDER, "converted.wav")
        hidden_wav_path = os.path.join(UPLOAD_FOLDER, "hidden.wav")
        final_audio_path = os.path.join(UPLOAD_FOLDER, "hidden" + ".wav")  # Toujours renvoyer en WAV

        file.save(original_path)

        # Conversion MP3 → WAV si nécessaire
        if original_ext == ".mp3":
            sound = AudioSegment.from_mp3(original_path)
            sound.export(wav_path, format="wav")
        else:
            wav_path = original_path

        # Cacher le message dans le fichier WAV
        hide_message(wav_path, message, key, hidden_wav_path)

        # Le fichier final sera toujours un WAV
        os.rename(hidden_wav_path, final_audio_path)

        return send_file(final_audio_path, as_attachment=True)

    except Exception as e:
        flash(f"Une erreur est survenue lors de l'encodage : {str(e)}")
        return redirect(url_for('encode_page'))

# Route POST pour décoder (extraire un message depuis l'audio)
@app.route('/decode', methods=['POST'])
def decode():
    try:
        file = request.files['audio']
        key = request.form['key'].ljust(16)[:16].encode()

        # Enregistrer le fichier audio
        original_ext = os.path.splitext(file.filename)[1].lower()
        original_path = os.path.join(UPLOAD_FOLDER, file.filename)
        wav_path = os.path.join(UPLOAD_FOLDER, "to_decode.wav")

        file.save(original_path)

        # Si le fichier est MP3, le convertir en WAV
        if original_ext == ".mp3":
            sound = AudioSegment.from_mp3(original_path)
            sound.export(wav_path, format="wav")
        else:
            wav_path = original_path

        # Extraire le message caché
        message = extract_message(wav_path, key)

        if message is None or message.strip() == "":
            flash("❌ Message introuvable ou mot de passe incorrect.")
            return redirect(url_for('decode_page'))

        flash("✅ Message extrait avec succès !")
        return render_template("steg-audio/decode.html", message=message)

    except Exception as e:
        flash(f"Une erreur est survenue lors du décodage : {str(e)}")
        return redirect(url_for('decode_page'))

if __name__ == '__main__':
    app.run(debug=True)

