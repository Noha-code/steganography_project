from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify 
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
import piexif
import piexif.helper
import os
import datetime
import threading
import webbrowser
import secrets
import logging
import requests
import json
import uuid
import random
import base64
import hashlib

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Rate limiting to prevent abuse
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

# Configuration for folders and files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "secured_images")
HONEYPOT_FOLDER = os.path.join(BASE_DIR, "tracking_endpoints")  # Renamed to be less obvious
LOG_FILE = os.path.join(BASE_DIR, "alert_log.txt")
ACCESS_LOG_FILE = os.path.join(BASE_DIR, "access_log.txt")
TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HONEYPOT_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "img"), exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "app.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Attractive file names for honeypots
ATTRACTIVE_FILENAMES = [
    "passwords", "admin_access", "confidential_data", "secret_keys", 
    "employee_salaries", "company_secrets", "server_credentials",
    "database_login", "vpn_access", "ssh_keys", "admin_passwords",
    "financial_report", "customer_data", "credit_cards", "personal_info"
]

# URL paths that look legitimate (rather than obvious "honeypot")
LEGITIMATE_URL_PATHS = [
    "financial-data", "secure-documents", "private-content", "access-portal",
    "restricted-files", "user-data", "account-info", "confidential-reports",
    "internal-docs", "client-records", "company-files", "personnel-records"
]

# Webhook configuration (to be defined by the user in the interface)
WEBHOOK_URL = None

# Function to create a unique tracking ID for each image
def generate_tracking_id():
    return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]

# Function to create a honeypot HTML file that will ping back when viewed
def create_honeypot_html(tracking_id):
    honeypot_path = os.path.join(HONEYPOT_FOLDER, f"{tracking_id}.html")
    
    # Choose a legitimate-looking URL path
    legitimate_path = random.choice(LEGITIMATE_URL_PATHS)
    
    # Create a simple HTML file with JavaScript that will ping back to the server
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Loading Secure Content...</title>
        <script>
            // Send ping to server immediately
            fetch('/track/{tracking_id}', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    userAgent: navigator.userAgent,
                    language: navigator.language,
                    screenSize: window.screen.width + 'x' + window.screen.height,
                    referrer: document.referrer
                }})
            }});

            // Redirect to a generic page after sending data
            setTimeout(() => {{
                window.location = "/not-found";
            }}, 500);
        </script>
    </head>
    <body>
        <p>Loading secure content. Please wait...</p>
    </body>
    </html>
    """
    
    with open(honeypot_path, "w") as f:
        f.write(html_content)
    
    # Assure-toi que la création du honeypot est enregistrée dans les logs
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{now}] Honeypot HTML created: {tracking_id}.html | Path: /{legitimate_path}/{tracking_id}\n"
    
    # S'assurer que le fichier de logs existe
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("")
    
    # Ajouter l'entrée au fichier de logs
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_message)
    
    return f"/{legitimate_path}/{tracking_id}"

# Function to embed tracking URL in image EXIF data
def embed_tracking_url(image_path, tracking_id, base_url):
    try:
        # Generate the full tracking URL with legitimate-looking path
        legitimate_path = random.choice(LEGITIMATE_URL_PATHS)
        tracking_url = f"{base_url}/{legitimate_path}/{tracking_id}"
        
        # Load existing EXIF data or create new
        try:
            exif_dict = piexif.load(image_path)
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # Create a comment with the tracking URL
        user_comment = piexif.helper.UserComment.dump(
            f"View additional information: {tracking_url}"
        )
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        
        # Add other tempting metadata
        exif_dict["0th"][piexif.ImageIFD.Copyright] = f"Confidential - {tracking_url}".encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.XPComment] = f"Secure link: {tracking_url}".encode('utf-16le')
        exif_dict["0th"][piexif.ImageIFD.DocumentName] = f"Protected content: Access via {tracking_url}".encode('utf-8')
        
        # Add a custom XMP metadata block with the tracking URL
        xmp_data = f"""
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                <rdf:Description rdf:about=""
                    xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:description>This image contains secure content. Access via: {tracking_url}</dc:description>
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        """.encode('utf-8')
        
        # If using XMP, you'd normally add it as a separate APP1 segment
        # For simplicity, we'll add the URL in multiple places
        
        # Save the EXIF data back to the image
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        
        # Additional metadata insertion for specific image types
        img = Image.open(image_path)
        if img.format == "PNG":
            # For PNG files, also add to the textual metadata
            img_with_data = img.copy()
            img_with_data.info["Description"] = f"Protected content. Access via: {tracking_url}"
            img_with_data.save(image_path)
        
        return True
    except Exception as e:
        logging.error(f"Error embedding tracking URL: {str(e)}")
        return False

# Function to add additional attractive metadata to the image
def enhance_image_for_attackers(image_path, title=None, keywords=None):
    try:
        # Default tempting metadata
        if not title:
            title = random.choice([
                "Confidential Company Information", 
                "Private Access Keys",
                "Personal Financial Records",
                "Employee Login Credentials",
                "Internal Server Configuration"
            ])
        
        if not keywords:
            keywords = "confidential,secret,private,secure,credentials,passwords,admin"
        
        # Load existing EXIF data or create new
        try:
            exif_dict = piexif.load(image_path)
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        # Add tempting metadata
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = title.encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.Software] = "SecureDoc Confidential".encode('utf-8')
        exif_dict["0th"][piexif.ImageIFD.Artist] = "Internal Use Only".encode('utf-8')
        
        # Add creation date as today (makes it seem current and important)
        now = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["0th"][piexif.ImageIFD.DateTime] = now.encode('utf-8')
        
        # Save the EXIF data back to the image
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        
        return True
    except Exception as e:
        logging.error(f"Error enhancing image: {str(e)}")
        return False

# Function to log access attempts
def log_access(tracking_id, ip, user_agent, additional_info=None):
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Additional information to include in the log
        info_str = ""
        if additional_info:
            info_str = f" | Additional Info: {additional_info}"
        
        log_message = f"[{now}] ACCESS DETECTED: Tracking ID {tracking_id} | IP: {ip} | User-Agent: {user_agent}{info_str}\n"
        
        # Create log file if it doesn't exist
        if not os.path.exists(ACCESS_LOG_FILE):
            with open(ACCESS_LOG_FILE, "w") as f:
                f.write("")
        
        with open(ACCESS_LOG_FILE, "a") as f:
            f.write(log_message)
        
        logging.info(f"Access recorded for tracking ID {tracking_id} from {ip}")
        
        # Send webhook alert if configured
        if WEBHOOK_URL:
            send_webhook_alert(log_message)
        
        return True
    except Exception as e:
        logging.error(f"Error in log_access: {str(e)}")
        return False

# Function to log image creation - Améliorée avec plus de détails
def log_creation(tracking_id, original_filename, ip, safe_filename=None, url_path=None):
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ajouter le nom de fichier sécurisé dans le log si disponible
        file_info = original_filename
        if safe_filename:
            file_info = f"{original_filename} → {safe_filename}"
            
        # Ajouter l'URL du honeypot si disponible
        path_info = ""
        if url_path:
            path_info = f" | Honeypot URL: {url_path}"
        
        log_message = f"[{now}] Image protected: {file_info} | Tracking ID: {tracking_id} | Created from IP: {ip}{path_info}\n"
        
        # Create log file if it doesn't exist
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                f.write("")
        
        with open(LOG_FILE, "a") as f:
            f.write(log_message)
        
        logging.info(f"Created protected image with tracking ID {tracking_id}")
        return True
    except Exception as e:
        logging.error(f"Error in log_creation: {str(e)}")
        return False

# Function to send webhook alerts
def send_webhook_alert(message):
    try:
        if not WEBHOOK_URL:
            return False
        
        # Format for Discord/Slack
        payload = {
            "content": f"🚨 *SECURITY ALERT* 🚨\n{message}",  # Changed name to be less obvious
            "username": "Security Monitoring System"  # Changed name to be less obvious
        }
        
        # Send the request
        response = requests.post(
            WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 204 or response.status_code == 200:
            logging.info("Webhook sent successfully")
            return True
        else:
            logging.error(f"Failed to send webhook: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error in send_webhook_alert: {str(e)}")
        return False

# Read logs for displaying history - Améliorée
def read_logs(log_type="creation"):
    try:
        # Assurez-vous d'utiliser le bon fichier de logs selon le type
        log_file = LOG_FILE if log_type == "creation" else ACCESS_LOG_FILE
        
        if not os.path.exists(log_file):
            return "No logs available."
            
        with open(log_file, "r") as f:
            log_content = f.read()
            
        # Si le fichier existe mais est vide
        if not log_content.strip():
            return "No logs available."
            
        return log_content
    except Exception as e:
        logging.error(f"Error reading logs: {str(e)}")
        return "Error reading logs."

# Route for homepage - Modified to serve index.html directly
@app.route('/honeyfiles')
def home():
    return render_template("honeyfiles/index.html")

# Route for main image protection page - Améliorée pour la création de honeypot
@app.route('/protect', methods=['GET', 'POST'])
@limiter.limit("30 per hour")  # Limit to prevent abuse
def protect_image():
    global WEBHOOK_URL
    
    if request.method == 'POST':
        try:
            # Update webhook URL if specified
            if 'webhook_url' in request.form and request.form['webhook_url'].strip():
                WEBHOOK_URL = request.form['webhook_url'].strip()
                
            # Check if a file is selected
            if 'image' not in request.files:
                flash("No file selected.", "error")
                return redirect(request.url)
                
            uploaded_file = request.files['image']
            if uploaded_file.filename == "":
                flash("No file selected.", "error")
                return redirect(request.url)
            
            # Check file type
            if not uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                flash("Unsupported file type. Use images (PNG, JPG, JPEG, BMP, GIF).", "error")
                return redirect(request.url)
            
            ip = request.remote_addr
            
            # Get or generate an attractive filename
            custom_filename = request.form.get('custom_filename', '')
            use_attractive_name = 'use_attractive_name' in request.form
            
            if custom_filename:
                file_base = custom_filename
            elif use_attractive_name:
                file_base = random.choice(ATTRACTIVE_FILENAMES)
            else:
                file_base = f"secure_image_{uuid.uuid4().hex[:8]}"
            
            # Get the extension from the original file
            _, file_ext = os.path.splitext(uploaded_file.filename)
            
            # Create secure filename
            safe_filename = f"{file_base}{file_ext}"
            image_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            
            # Save the original image
            uploaded_file.save(image_path)
            
            # Generate a tracking ID for this image
            tracking_id = generate_tracking_id()
            
            # Create honeypot HTML file that will track when accessed
            honeypot_relative_url = create_honeypot_html(tracking_id)
            
            # Get base URL for tracking link
            base_url = request.url_root.rstrip('/')
            honeypot_full_url = f"{base_url}{honeypot_relative_url}"
            
            # Add tempting metadata based on form inputs
            title = request.form.get('metadata_title', '')
            keywords = request.form.get('metadata_keywords', '')
            enhance_image_for_attackers(image_path, title, keywords)
            
            # Embed tracking URL in EXIF data
            if embed_tracking_url(image_path, tracking_id, base_url):
                # Log the creation avec le nom de fichier sécurisé et l'URL du honeypot
                log_creation(tracking_id, uploaded_file.filename, ip, safe_filename, honeypot_full_url)
                
                flash("Image protected successfully!", "success")
                flash(f"Your image now contains hidden tracking links. If someone accesses this image and views its metadata, their access will be detected.", "info")
                
                # Offer the download
                return send_file(image_path, as_attachment=True, download_name=safe_filename)
            else:
                flash("Error while protecting the image.", "error")
                return redirect(request.url)
        
        except Exception as e:
            logging.error(f"Error in protect_image POST route: {str(e)}")
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(request.url)

    # GET method - Display page with logs
    creation_logs = read_logs("creation")  # Lit les logs de création
    access_logs = read_logs("access")      # Lit les logs d'accès
    return render_template("honeyfiles/protect_image.html", creation_logs=creation_logs, access_logs=access_logs, webhook_url=WEBHOOK_URL)

# Routes for legitimate-looking paths that actually serve the honeypot
@app.route('/financial-data/<tracking_id>')
@app.route('/secure-documents/<tracking_id>')
@app.route('/private-content/<tracking_id>')
@app.route('/access-portal/<tracking_id>')
@app.route('/restricted-files/<tracking_id>')
@app.route('/user-data/<tracking_id>')
@app.route('/account-info/<tracking_id>')
@app.route('/confidential-reports/<tracking_id>')
@app.route('/internal-docs/<tracking_id>')
@app.route('/client-records/<tracking_id>')
@app.route('/company-files/<tracking_id>')
@app.route('/personnel-records/<tracking_id>')
def serve_honeypot(tracking_id):
    try:
        # Serve the honeypot HTML file
        honeypot_path = os.path.join(HONEYPOT_FOLDER, f"{tracking_id}.html")
        if os.path.exists(honeypot_path):
            return send_file(honeypot_path)
        else:
            return redirect(url_for('not_found'))
    except Exception as e:
        logging.error(f"Error serving honeypot: {str(e)}")
        return redirect(url_for('not_found'))

# Route to track accesses to honeypot links
@app.route('/track/<tracking_id>', methods=['POST'])
def track_access(tracking_id):
    try:
        # Get information about the request
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Get additional information from the POST data
        additional_info = ""
        if request.is_json:
            data = request.get_json()
            referrer = data.get('referrer', 'Unknown')
            language = data.get('language', 'Unknown')
            screen_size = data.get('screenSize', 'Unknown')
            additional_info = f"Referrer: {referrer} | Language: {language} | Screen: {screen_size}"
        
        # Log the access dans le bon fichier
        log_access(tracking_id, ip, user_agent, additional_info)
        
        return jsonify({"status": "logged"})
    except Exception as e:
        logging.error(f"Error in track_access: {str(e)}")
        return jsonify({"status": "error"})

# API route to check for new accesses
@app.route('/api/check-accesses', methods=['GET'])
def check_accesses():
    log_type = request.args.get('log_type', 'access')
    last_access = request.args.get('last_timestamp', '')
    logs = read_logs(log_type)  # Peut récupérer soit les logs d'accès, soit les logs de création
    
    # Simple logic: check if logs contain new entries since last check
    has_new = False
    if last_access and logs and logs != "No logs available." and logs != "Error reading logs.":
        lines = logs.strip().split('\n')
        if lines and lines[-1].startswith(f"[{last_access}]"):
            has_new = False
        else:
            has_new = True
    
    return jsonify({
        "has_new_logs": has_new,
        "full_log": logs
    })

# 404 error handler - Updated for consistent template use
@app.errorhandler(404)
@app.route('/not-found')
def not_found(e=None):
    return render_template("honeyfiles/404.html"), 404

# Route for quick image protection with default parameters - Améliorée
@app.route('/quick-protect', methods=['POST'])
def quick_protect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file selected"}), 400
                
        uploaded_file = request.files['image']
        if uploaded_file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Use a random attractive filename
        file_base = random.choice(ATTRACTIVE_FILENAMES)
        _, file_ext = os.path.splitext(uploaded_file.filename)
        safe_filename = f"{file_base}{file_ext}"
        image_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        # Save the image
        uploaded_file.save(image_path)
        
        # Generate tracking ID
        tracking_id = generate_tracking_id()
        
        # Create honeypot HTML and get its URL
        honeypot_relative_url = create_honeypot_html(tracking_id)
        base_url = request.url_root.rstrip('/')
        honeypot_full_url = f"{base_url}{honeypot_relative_url}"
        
        # Add tempting metadata
        enhance_image_for_attackers(image_path)
        
        # Embed tracking URL
        if embed_tracking_url(image_path, tracking_id, base_url):
            # Log creation avec l'URL complète du honeypot
            log_creation(tracking_id, uploaded_file.filename, request.remote_addr, safe_filename, honeypot_full_url)
            
            # Generate access URL
            access_url = url_for('serve_protected_image', filename=safe_filename, _external=True)
            
            return jsonify({
                "success": True,
                "filename": safe_filename,
                "access_url": access_url
            })
        else:
            return jsonify({"error": "Failed to protect image"}), 500
    
    except Exception as e:
        logging.error(f"Error in quick_protect: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route to serve protected images
@app.route('/protected/<filename>')
def serve_protected_image(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        logging.error(f"Error serving protected image: {str(e)}")
        return "Error accessing file", 500

# Automatically open browser on the correct page
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")  # Updated to use the home page

if __name__ == '__main__':
    # Open browser after a delay to let the server start
    threading.Timer(1.5, open_browser).start()
    
    # Start the application
    app.run(debug=True, host='0.0.0.0', port=5000)
