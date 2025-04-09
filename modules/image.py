from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from Cryptodome.Cipher import AES
import base64

app = Flask(__name__)
app.secret_key = 'secretkey123'  # Nécessaire pour utiliser flash()

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la page de l'encodage (cacher un message)
@app.route('/encode')
def encode_page():
    return render_template('encode.html')

# Route pour la page du décodage (extraire un message)
@app.route('/decode')
def decode_page():
    return render_template('decode.html')

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

# Route POST pour encoder
@app.route('/encode', methods=['POST'])
def encode():
    try:
        file = request.files['image']
        message = request.form['message']
        key = request.form['key'].ljust(16)[:16].encode()

        img = Image.open(file).convert("RGB")
        encrypted_message = encrypt_message(message, key) + "###"
        binary_message = ''.join(format(ord(c), '08b') for c in encrypted_message)

        pixels = img.load()
        data_index = 0

        for y in range(img.height):
            for x in range(img.width):
                r, g, b = pixels[x, y]
                if data_index < len(binary_message):
                    r = (r & 0xFE) | int(binary_message[data_index])
                    data_index += 1
                if data_index < len(binary_message):
                    g = (g & 0xFE) | int(binary_message[data_index])
                    data_index += 1
                if data_index < len(binary_message):
                    b = (b & 0xFE) | int(binary_message[data_index])
                    data_index += 1
                pixels[x, y] = (r, g, b)
                if data_index >= len(binary_message):
                    break
            if data_index >= len(binary_message):
                break

        output = BytesIO()
        img.save(output, format="PNG")
        output.seek(0)

        return send_file(output, mimetype='image/png', as_attachment=True, download_name='stegano.png')

    except UnidentifiedImageError:
        flash("Invalid image format.")
    except Exception as e:
        flash(f"An error occurred during encoding: {str(e)}")
    return redirect(url_for('encode_page'))

# Route POST pour décoder
@app.route('/decode', methods=['POST'])
def decode():
    try:
        # Récupération de l'image et de la clé
        file = request.files['image']
        key = request.form['key'].ljust(16)[:16].encode()

        # Ouverture de l'image
        img = Image.open(file).convert("RGB")
        pixels = img.load()
        binary_message = ""
        
        # Extraction du message binaire des pixels avec limite de taille raisonnable
        # et vérification du séparateur
        max_bits = 100000  # Limite raisonnable pour éviter une boucle infinie
        found_separator = False
        
        for y in range(img.height):
            if found_separator or len(binary_message) > max_bits:
                break
                
            for x in range(img.width):
                r, g, b = pixels[x, y]
                binary_message += str(r & 1)
                binary_message += str(g & 1)
                binary_message += str(b & 1)
                
                # Vérification si nous avons assez de bits pour faire une vérification de séparateur
                if len(binary_message) >= 24:  # Au moins 3 caractères (24 bits)
                    # Décodons temporairement pour voir si nous avons atteint "###"
                    temp_chars = [binary_message[i:i+8] for i in range(0, len(binary_message), 8)]
                    temp_message = ''.join(chr(int(char, 2)) for char in temp_chars if len(char) == 8)
                    
                    if "###" in temp_message:
                        found_separator = True
                        break
                        
                if len(binary_message) > max_bits:
                    break
        
        # Décodage du message binaire
        chars = [binary_message[i:i+8] for i in range(0, len(binary_message), 8) if len(binary_message[i:i+8]) == 8]
        secret_message = ''.join(chr(int(char, 2)) for char in chars)
        
        # Extraction du message avant le séparateur
        if "###" in secret_message:
            secret_message = secret_message.split("###")[0]
        else:
            flash("No hidden message found or incomplete message.")
            return redirect(url_for('decode_page'))

        # Décryptage du message
        message = decrypt_message(secret_message, key)
        if message is None:
            flash("Incorrect password or corrupted image.")
            return redirect(url_for('decode_page'))

        return render_template("decode.html", message=message)

    except UnidentifiedImageError:
        flash("Invalid image format.")
    except Exception as e:
        flash(f"An error occurred during decoding: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Pour déboguer
    return redirect(url_for('decode_page'))


if __name__ == '__main__':
    app.run(debug=True)
