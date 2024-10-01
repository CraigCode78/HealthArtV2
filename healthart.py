import os
from flask import Flask, redirect, request, session, jsonify, url_for, render_template, abort, send_file
from requests_oauthlib import OAuth2Session
import requests
import logging
from openai import OpenAI, OpenAIError, APIError, APIConnectionError, RateLimitError
import base64
from threading import Thread
import time
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import json
import io
from replit import db
from replit.object_storage import Client as ObjectStorageClient


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')

# The bucket name you provided
BUCKET_NAME = os.environ.get('REPLIT_OBJECT_STORE_BUCKET', 'default-bucket')

# Allow OAuth over HTTP for development purposes only
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# WHOOP API Configuration
CLIENT_ID = os.environ.get('WHOOP_CLIENT_ID')  # WHOOP OAuth2 Client ID from environment variable
CLIENT_SECRET = os.environ.get('WHOOP_CLIENT_SECRET')  # WHOOP OAuth2 Client Secret from environment variable
REDIRECT_URI = 'https://healthartv2-craighepburn.replit.app/callback'  # OAuth2 Redirect URI
AUTH_URL = 'https://api.prod.whoop.com/oauth/oauth2/auth'  # WHOOP OAuth2 Authorization URL
TOKEN_URL = 'https://api.prod.whoop.com/oauth/oauth2/token'  # WHOOP OAuth2 Token URL
API_BASE_URL = 'https://api.prod.whoop.com/developer'  # WHOOP API Base URL

# Initialize OpenAI client with API key from environment variable
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configure logging to display debug information
logging.basicConfig(level=logging.DEBUG)

# Initialize the Object Storage client
object_storage_client = ObjectStorageClient()
BUCKET_NAME = os.environ.get('REPLIT_OBJECT_STORE_BUCKET', 'default-bucket')

# Remove SQLAlchemy setup and GalleryImage model

def token_updater(token):
    """
    Updates the OAuth token in the user session.

    Args:
        token (dict): The OAuth token data.
    """
    session['oauth_token'] = token

def get_whoop_session():
    """
    Retrieves the OAuth2 session for WHOOP API interactions.

    Returns:
        OAuth2Session or None: Returns an OAuth2Session object if token exists, else None.
    """
    token = session.get('oauth_token')
    if not token:
        return None
    return OAuth2Session(
        CLIENT_ID,
        token=token,
        auto_refresh_url=TOKEN_URL,
        auto_refresh_kwargs={
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        },
        token_updater=token_updater
    )

@app.route('/')
def index():
    """
    Renders the homepage of the HealthArt application.

    Returns:
        Rendered HTML template for the index page.
    """
    return render_template('index.html')

@app.route('/login')
def login():
    """
    Initiates the OAuth2 login flow with WHOOP.

    Creates an OAuth2Session and redirects the user to the WHOOP authorization URL.

    Returns:
        Redirect: Redirects the user to WHOOP's authorization page.
    """
    whoop = OAuth2Session(
        CLIENT_ID,
        redirect_uri=REDIRECT_URI, 
        scope=['read:profile', 'read:recovery', 'read:workout', 'read:sleep']
    )
    authorization_url, state = whoop.authorization_url(AUTH_URL)
    session['oauth_state'] = state  # Save state to prevent CSRF
    logging.info(f"Authorization URL: {authorization_url}")
    logging.info(f"State: {state}")
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    """
    Handles the OAuth2 callback from WHOOP after user authorization.

    Exchanges the authorization code for an access token and updates the session.

    Returns:
        Redirect: Redirects the user to the health_art page upon successful authentication.
        JSON Response: Returns an error message if authentication fails.
    """
    whoop = OAuth2Session(
        CLIENT_ID,
        state=session.get('oauth_state'),
        redirect_uri=REDIRECT_URI
    )
    try:
        # Fetch the OAuth2 token using the authorization response
        token = whoop.fetch_token(
            TOKEN_URL,
            authorization_response=request.url,
            include_client_id=True,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            method='POST',
            auth=None,  # Disable client_secret_basic
            state=session.get('oauth_state')
        )
    except Exception as e:
        logging.error(f"Error fetching token: {e}")
        return jsonify({"error": "Authentication failed"}), 500

    token_updater(token)  # Update the session with the new token
    return redirect(url_for('health_art'))

def generate_ai_art(recovery_score, additional_metrics=None):
    """
    Generates abstract AI art based on health data using OpenAI's DALL-E.

    Args:
        recovery_score (float): The primary recovery score (1-99).
        additional_metrics (dict, optional): Additional health metrics like sleep quality, strain, HRV.

    Returns:
        str or None: Base64-encoded image data if successful, else None.
    """
    # Base prompt structure for DALL-E
    base_prompt = "Create an abstract digital artwork representing health data with the following elements:"

    # Determine color scheme based on recovery score
    if 1 <= recovery_score <= 33:
        color_prompt = f"Dominant colors are vibrant reds, representing low recovery ({recovery_score}% recovery score)."
    elif 34 <= recovery_score <= 66:
        color_prompt = f"Mix of warm amber and cool tones, balancing moderate recovery ({recovery_score}% recovery score)."
    elif 67 <= recovery_score <= 99:
        color_prompt = f"Dominant colors are vibrant greens and blues, representing high recovery ({recovery_score}% recovery score)."
    else:
        color_prompt = f"Colors are neutral, indicating an undefined recovery score ({recovery_score}%)."

    # Define pattern and shape elements based on recovery score
    pattern_prompt = [
        "Incorporate flowing, organic shapes to represent flexibility and adaptability.",
        "Use repeating geometric patterns, with their regularity affected by the recovery score.",
        f"{'Dense' if recovery_score > 66 else 'Sparse'} network of interconnected lines symbolizing bodily systems.",
        f"Abstract {'circular' if recovery_score > 60 else 'angular'} forms representing energy levels."
    ]

    # Additional metric representations if provided
    metric_prompts = []
    if additional_metrics:
        if 'sleep_quality' in additional_metrics:
            sleep_quality = additional_metrics['sleep_quality']
            metric_prompts.append(f"Represent sleep quality ({sleep_quality}%) with {'smooth' if sleep_quality > 70 else 'jagged'} wave-like patterns.")
        if 'strain' in additional_metrics:
            strain = additional_metrics['strain']
            metric_prompts.append(f"Illustrate physical strain ({strain}/21) with {'bold' if strain > 15 else 'subtle'} textural elements.")
        if 'hrv' in additional_metrics:
            hrv = additional_metrics['hrv']
            metric_prompts.append(f"Depict heart rate variability ({hrv} ms) using {'intricate' if hrv > 50 else 'simple'} fractal-like structures.")

    # Combine all prompt elements into a final prompt for DALL-E
    final_prompt = f"{base_prompt} {color_prompt} {' '.join(pattern_prompt)} {' '.join(metric_prompts)} The overall composition should be harmonious yet dynamic, clearly reflecting the health status through abstract visual elements."

    # Use OpenAI's DALL-E to generate the art based on the prompt
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            n=1,
            size="1024x1024",
            quality="standard",
            response_format="b64_json"
        )
        logging.debug(f"OpenAI response: {response}")
        image_data = response.data[0].b64_json  # Extract base64 image data
        return image_data
    except Exception as e:
        logging.error(f"Error generating art: {str(e)}")
        return None

@app.route('/health_art')
def health_art():
    """
    Fetches the user's recovery data from WHOOP and renders the health_art page.

    Retrieves the recovery score and passes it to the health_art.html template.

    Returns:
        Rendered HTML template for the health_art page.
        JSON Response: Returns an error message if data retrieval fails.
    """
    whoop = get_whoop_session()
    if not whoop:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

    try:
        # Fetch recovery data from WHOOP API
        recovery_resp = whoop.get(f"{API_BASE_URL}/v1/recovery")
        recovery_resp.raise_for_status()
        recovery_data = recovery_resp.json()
        logging.debug(f"Recovery data: {recovery_data}")
    except Exception as e:
        logging.error(f"Error fetching recovery data: {str(e)}")
        return jsonify({"error": "Failed to fetch recovery data"}), 500

    try:
        # Extract the recovery score from the response and cast to integer
        recovery_score = int(recovery_data['records'][0]['score']['recovery_score'])
        logging.debug(f"Recovery Score: {recovery_score}")
    except (KeyError, IndexError, ValueError) as e:
        logging.error(f"Error extracting recovery score: {str(e)}")
        return jsonify({"error": "Failed to extract recovery score"}), 500

    # Render the health_art.html template with the recovery score
    return render_template('health_art.html', recovery_score=recovery_score)

@app.route('/gallery')
def gallery():
    try:
        # List all objects in the bucket
        all_objects = object_storage_client.list(BUCKET_NAME)
        
        # Filter for image files (assuming all are .png)
        image_files = [obj for obj in all_objects if obj.startswith("health_art_") and obj.endswith(".png")]
        
        # Generate URLs for the frontend to access the images
        image_urls = [f"/static/gallery/{image}" for image in image_files]
        
        return render_template('gallery.html', images=image_urls)
    except Exception as e:
        logging.error(f"Error fetching gallery: {str(e)}")
        return jsonify({"error": f"Failed to fetch gallery: {str(e)}"}), 500

@app.route('/generate_art', methods=['POST'])
def generate_art():
    whoop = get_whoop_session()
    if not whoop:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        recovery_resp = whoop.get(f"{API_BASE_URL}/v1/recovery")
        recovery_resp.raise_for_status()
        recovery_data = recovery_resp.json()
        recovery_score = int(recovery_data['records'][0]['score']['recovery_score'])
    except Exception as e:
        logging.error(f"Error fetching recovery data: {str(e)}")
        return jsonify({"error": "Failed to fetch recovery data"}), 500

    art_base64 = generate_ai_art(recovery_score)
    if art_base64 is None:
        return jsonify({"error": "Failed to generate AI art"}), 500

    try:
        image_filename = f"health_art_{uuid.uuid4()}.png"
        image_bytes = base64.b64decode(art_base64)
        
        # Save to Replit's Object Storage
        object_storage_client.upload_from_bytes(f"{BUCKET_NAME}/{image_filename}", image_bytes)
        logging.debug(f"Saved image to storage: {image_filename}")
        
        image_url = url_for('serve_image', filename=image_filename, _external=True)
    except Exception as e:
        logging.error(f"Error saving image to storage: {str(e)}")
        return jsonify({"error": "Failed to save generated image"}), 500

    return jsonify({
        "image_data": art_base64,
        "image_url": image_url
    })

UPLOAD_FOLDER = 'static/gallery'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save_image', methods=['POST'])
def save_image():
    if 'file' not in request.files:
        logging.warning("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    recovery_score = request.form.get('recovery_score')
    logging.debug(f"Recovery score received: {recovery_score}")
    if file.filename == '':
        logging.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{recovery_score}.png")
        try:
            # Upload image to Object Storage
            object_storage_client.upload_from_bytes(filename, file.read())
            logging.debug(f"Uploaded image to Object Storage: {filename}")
            
            # Save metadata
            metadata = {
                'filename': filename,
                'recovery_score': recovery_score,
                'timestamp': datetime.now().isoformat()
            }
            metadata_filename = f"{filename}_metadata.json"
            object_storage_client.upload_from_bytes(metadata_filename, json.dumps(metadata).encode())
            logging.debug(f"Uploaded metadata to Object Storage: {metadata_filename}")
            
            return jsonify({"message": "File saved successfully", "metadata": metadata}), 200
        except Exception as e:
            logging.error(f"Error saving image and metadata: {e}")
            return jsonify({"error": "Internal server error"}), 500
    logging.warning("File type not allowed")
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/get_image/<filename>')
def get_image(filename):
    try:
        image = object_storage_client.download_as_bytes(filename)
        logging.debug(f"Downloaded image from Object Storage: {filename}")
        return send_file(
            io.BytesIO(image),
            mimetype='image/png',
            as_attachment=False,
            download_name=filename
        )
    except Exception as e:
        logging.error(f"Error retrieving image {filename}: {e}")
        abort(404)

@app.route('/image/<filename>')
def image_detail(filename):
    try:
        metadata_filename = f"{filename}_metadata.json"
        metadata = json.loads(object_storage_client.download_as_bytes(metadata_filename).decode())
        logging.debug(f"Fetched metadata for image {filename}")
        return render_template('image_detail.html', image=metadata)
    except Exception as e:
        logging.error(f"Error retrieving image detail for {filename}: {e}")
        abort(404)

@app.route('/favicon.ico')
def favicon():
    """
    Redirects requests for favicon to the static favicon.ico file.

    Returns:
        Redirect: Redirects to the favicon.ico file in the static directory.
    """
    return redirect(url_for('static', filename='favicon.ico'))

@app.errorhandler(404)
def page_not_found(e):
    """
    Handles 404 Page Not Found errors.

    Logs the error and returns a JSON response indicating the page was not found.

    Args:
        e (Exception): The exception that was raised.

    Returns:
        JSON Response: Error message with 404 status code.
    """
    logging.error(f"Page not found: {e}")
    return jsonify(error="Page not found"), 404

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An unexpected error occurred: {e}")
    return jsonify(error=str(e)), 500

GALLERY_DIR = os.path.join(app.static_folder, 'gallery')
os.makedirs(GALLERY_DIR, exist_ok=True)

@app.route('/serve_image/<path:filename>')
def serve_image(filename):
    try:
        # Prepend "default-bucket/" if it's not already there
        if not filename.startswith("default-bucket/"):
            filename = f"default-bucket/{filename}"
        
        # Download the image from Replit's Object Storage
        image_bytes = object_storage_client.download_as_bytes(filename)
        
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/png',
            as_attachment=False,
            download_name=filename.split('/')[-1]
        )
    except Exception as e:
        logging.error(f"Error serving image {filename}: {str(e)}")
        return "Image not found", 404

@app.route('/get_gallery_images')
def get_gallery_images():
    try:
        # List all objects in the default bucket
        all_objects = list(object_storage_client.list())
        logging.debug(f"All objects in bucket: {all_objects}")
        
        # Print out the type and length of all_objects
        logging.debug(f"Type of all_objects: {type(all_objects)}")
        logging.debug(f"Length of all_objects: {len(all_objects)}")
        
        # Filter for image files (assuming all are .png)
        image_files = [obj.name for obj in all_objects if obj.name.startswith("default-bucket/health_art_") and obj.name.endswith(".png")]
        logging.debug(f"Filtered image files: {image_files}")
        
        # Generate URLs for the frontend to access the images
        image_urls = [url_for('serve_image', filename=image.split('/')[-1], _external=True) for image in image_files]
        logging.debug(f"Image URLs: {image_urls}")
        
        return jsonify({"images": image_urls})
    except Exception as e:
        logging.error(f"Error fetching gallery images: {str(e)}")
        return jsonify({"error": f"Failed to fetch gallery images: {str(e)}"}), 500

@app.route('/list_all_objects')
def list_all_objects():
    try:
        all_objects = list(object_storage_client.list())
        return jsonify({"objects": all_objects})
    except Exception as e:
        logging.error(f"Error listing all objects: {str(e)}")
        return jsonify({"error": f"Failed to list all objects: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)