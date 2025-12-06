from flask import Flask, render_template, redirect, url_for, session, request, flash, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import io
import json
import time
import base64
import random
import string
import smtplib
import numpy as np
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, auth
import tensorflow as tf
from functools import wraps
from werkzeug.middleware.proxy_fix import ProxyFix 
import requests

# Define base directory and load .env
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = Flask(__name__)

# ==========================================
#  ENVIRONMENT CHECK
# ==========================================
# 1. Checks if running on Render (auto-detected)
# 2. Checks if FLASK_ENV is set to 'production' (For Hostinger/VPS)
IS_PRODUCTION = os.getenv('RENDER') is not None or os.getenv('FLASK_ENV') == 'production'

print("---------------------------------------------------------")
if IS_PRODUCTION:
    print(f"✅ RUNNING IN PRODUCTION MODE (Secure Cookies: ON)")
else:
    print(f"⚠️  RUNNING IN DEVELOPMENT MODE (Secure Cookies: OFF)")
print("---------------------------------------------------------")

# ==========================================
#  GEMINI AI & CHATBOT CONFIGURATION
# ==========================================

# 1. Load API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# 2. Define System Instructions
SYSTEM_INSTRUCTION_TEXT = """
You are "Mangosteenify", the expert assistant for the MangosQueen application.

YOUR STRICT KNOWLEDGE BASE:
1. Mangosteen Farming (Planting, Harvest, Health Benefits).
2. Mangosteen Diseases (Anthracnose, Dry Rot, Gummosis).
3. MangosQueen System Features (Scanning, Real-time Detection, Sorting).

STRICT RULES FOR ANSWERING:
1. LANGUAGE MATCHING (CRITICAL): 
   - You **MUST** detect the user's primary language.
   - If the user asks in **ENGLISH**, answer only in **ENGLISH**.
   - If the user asks in **TAGALOG**, answer only in **TAGALOG**.
   - If the user uses **TAGLISH**, answer in **TAGLISH**.
2. LENGTH: Keep answers **SHORT, CONCISE, and DIRECT** (Max 3-4 sentences). Use bullet points for steps/lists.
3. GENERAL TOPICS (New Rule): 
   - If the user asks a simple, harmless question (like asking for a joke, the weather, or a quick fact), you **MAY** answer briefly.
   - If the user asks a complex, political, or extended off-topic question, politely redirect them back to Mangosteen or the MangosQueen app.

SPECIFIC ANSWERS FOR APP USAGE:
If the user asks "How to use the app?", "Paano gamitin?", or about features, use this detailed logic:

(English Context):
"The MangosQueen system is designed for easy use:
1. **Register & Login** to your account.
2. Go to the **Dashboard** to initiate the **Scan** feature. You have three choices: **Upload Image**, **Camera Capture**, or **Real-time Detection**.
3. **Analysis:** The system detects fruit diseases.
4. **Sorting:** It also provides a **Fruit Sorting mechanism** based on **disease** and **ripeness** status.
5. **Guidance:** You can also check the **About Page** for a complete system guide."

(Tagalog Context):
"Ang sistema ng MangosQueen ay simple at madaling gamitin:
1. **Access:** Kailangan mo munang **Mag-Register** at pagkatapos ay **Mag-Login** sa iyong account.
2. **Scan/Detection:** Pumunta sa **Dashboard** para gamitin ang **Scan** feature. May tatlong paraan: **Upload Image**, **Gamitin ang Camera**, o **Real-time Detection**.
3. **Analysis:** Dito ay matutukoy ng sistema ang sakit ng prutas.
4. **Sorting:** Meron ding **Fruit Sorting mechanism** para i-kategorya ang prutas base sa **sakit** at **pagkahinog**.
5. **Guidance:** Maari ka ring pumunta sa **About Page** para sa kumpletong gabay sa sistema."
"""

# 3. Initialize Gemini Model
model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash", 
            system_instruction=SYSTEM_INSTRUCTION_TEXT, 
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 300, 
            }
        )
        print("✅ Gemini AI Model (gemini-2.0-flash) configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini model: {e}")
        model = None
else:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env file.")

# ==========================================
# END OF CONFIGURATION
# ==========================================

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-dev-key-if-env-fails')

# FIXED: Configure session to expire on browser close
app.config.update(
    # Only use Secure cookies (HTTPS) if in Production. False for Localhost (HTTP).
    SESSION_COOKIE_SECURE=IS_PRODUCTION, 
    SESSION_COOKIE_HTTPONLY=True,         # Prevent JavaScript access
    SESSION_COOKIE_SAMESITE='Lax',        # CSRF protection
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),  # Server-side session timeout
    SESSION_PERMANENT=False,              # CRITICAL: Make sessions non-persistent
)

# FIXED: Only use ProxyFix in Production (Render/Hostinger)
if IS_PRODUCTION:
    # x_proto=1 ensures Flask knows you are using HTTPS on Hostinger
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if not firebase_admin._apps:
    try:
        private_key = os.getenv('FIREBASE_PRIVATE_KEY')
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
        
        if private_key and project_id and client_email:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": private_key.replace('\\n', '\n'),
                "client_email": client_email,
                "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_X509_CERT_URL'),
                "universe_domain": "googleapis.com"
            })
            firebase_admin.initialize_app(cred)
            print("Firebase initialized successfully using Environment Variables.")
        else:
            print("CRITICAL ERROR: Firebase environment variables are missing from .env file.")
            
    except Exception as e:
        print(f"Failed to initialize Firebase: {str(e)}")

db = firestore.client()

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# -------------------- SESSION SECURITY DECORATOR --------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Check if user is logged in
        if 'logged_in' not in session or not session['logged_in']:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        
        # 2. Prevent Caching (Fixes Back Button Issue)
        response = make_response(f(*args, **kwargs))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return decorated_function

# -------------------- AUTHENTICATION ROUTES --------------------
@app.route('/')
def home():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required # <--- Protected
def dashboard():
    return render_template('dashboard.html', logged_in=True)


# -------------------- LOGIN ROUTES --------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        login_input = request.form['email'].strip()
        password = request.form['password']

        users_ref = db.collection('user')
        
        # 1. Try to find user by EMAIL (convert input to lowercase)
        query = users_ref.where('email', '==', login_input.lower()).limit(1)
        users = query.get()
        
        # 2. If not found by email, try to find by USERNAME (exact match)
        if len(users) == 0:
            query = users_ref.where('username', '==', login_input).limit(1)
            users = query.get()
        
        if len(users) == 1:
            user_doc = users[0]
            user = user_doc.to_dict()
            user['id'] = user_doc.id
            
            if user.get('password') is None:
                flash("This account uses Google sign-in. Please use 'Continue with Google' to login.", "warning")
                return render_template('login.html')
            
            if check_password_hash(user['password'], password):
                session['logged_in'] = True
                session['email'] = user['email']
                session['user_id'] = user['id']
                session['login_message_shown'] = False
                
                # CRITICAL: Force session modified to ensure cookie is sent
                session.modified = True 
                session.permanent = False
                
                users_ref.document(user['id']).update({
                    'datetime_login': datetime.now().isoformat()
                })
                
                flash("Logged In Successfully!", "success")
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid Email/Username or Password!", "danger")
        else:
            flash("Invalid Email/Username or Password!", "danger")
            
    return render_template('login.html')

# -------------------- LOGOUT ROUTES --------------------

@app.route('/logout')
def logout():
    print(f"Session user_id: {session.get('user_id')}")
    
    if session.get('user_id'):
        user_id = session['user_id']
        try:
            user_ref = db.collection('user').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_ref.update({
                    'datetime_logout': datetime.now().isoformat()
                })
        except Exception as e:
            print(f"Error updating logout time: {e}")
    
    session.clear() 
    flash("Logged Out Successfully", "info")
    return redirect(url_for('home'))


# -------------------- CORS HELPER FUNCTIONS --------------------

def add_cors_headers(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Add CORS headers to response
        origin = request.headers.get('Origin', '')
        allowed_origins = [
            'https://mangosqueen.onrender.com',  # Your actual Render domain
            'http://localhost:5000',
            'http://127.0.0.1:5000'
        ]
        
        response_headers = {}
        if origin in allowed_origins:
            response_headers = {
                'Access-Control-Allow-Origin': origin,
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Allow-Credentials': 'true'
            }
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = jsonify({'success': True})
            response.headers.extend(response_headers)
            return response
        
        # Call the actual route function
        response_obj = f(*args, **kwargs)
        
        # Add headers to the response
        if isinstance(response_obj, tuple):
            # Response is a tuple (data, status, headers)
            data, status, headers = response_obj
            if headers is None:
                headers = {}
            headers.update(response_headers)
            return data, status, headers
        else:
            # Response is just data (Response Object)
            # Ensure we don't overwrite existing headers if we can help it, 
            # but usually this returns a Response object
            if hasattr(response_obj, 'headers'):
                response_obj.headers.extend(response_headers)
                return response_obj
            return response_obj, 200, response_headers
    
    return decorated_function


# -------------------- CONTINUE WITH EMAIL ROUTES --------------------

@app.route('/google_login', methods=['POST', 'OPTIONS'])
@add_cors_headers
def google_login():
    try:
        data = request.get_json()
        if not data:
            print("❌ Google Login: No JSON data received")
            return jsonify({'success': False, 'message': 'No data received'}), 400
            
        id_token = data.get('id_token')
        email = data.get('email')
        name = data.get('name')
        photo_url = data.get('photo_url')

        print(f"🔐 Google login attempt for: {email}")
        
        if not id_token:
            print("❌ Google Login: No ID token provided")
            return jsonify({'success': False, 'message': 'No ID token provided'}), 400

        try:
            # Verify the token
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            print(f"✅ Token verified for UID: {uid}")
            
            token_email = decoded_token.get('email')
            if token_email and email and token_email.lower() != email.lower():
                return jsonify({'success': False, 'message': 'Email verification failed'}), 401
                
        except Exception as verify_error:
            print(f"❌ Token verification error: {verify_error}")
            return jsonify({'success': False, 'message': 'Invalid ID token'}), 401

        # Database operations
        users_ref = db.collection('user')
        query = users_ref.where('email', '==', email).limit(1)
        users = query.get()

        user_id = None
        
        try:
            if len(users) > 0:
                # Existing user
                user_doc = users[0]
                user_id = user_doc.id
                
                update_data = {
                    'datetime_login': datetime.now().isoformat(),
                    'auth_method': 'google'
                }
                
                if photo_url:
                    existing_user = user_doc.to_dict()
                    if existing_user.get('photo_url') != photo_url:
                        update_data['photo_url'] = photo_url
                
                users_ref.document(user_id).update(update_data)
                
            else:
                # New user
                username_base = email.split('@')[0]
                username = username_base
                counter = 1
                while True:
                    username_query = users_ref.where('username', '==', username).limit(1).get()
                    if len(username_query) == 0:
                        break
                    username = f"{username_base}{counter}"
                    counter += 1
                
                new_user = {
                    'name': name or 'Google User',
                    'username': username,
                    'email': email,
                    'password': None,
                    'photo_url': photo_url,
                    'provider': 'google',
                    'firebase_uid': uid,
                    'auth_method': 'google',
                    'datetime_login': datetime.now().isoformat(),
                    'datetime_logout': None,
                    'created_at': datetime.now().isoformat()
                }
                doc_ref = users_ref.add(new_user)
                user_id = doc_ref[1].id
                
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            return jsonify({'success': False, 'message': 'Database error'}), 500

        # Session management
        try:
            session['logged_in'] = True
            session['username'] = email.split('@')[0]
            session['email'] = email
            session['user_id'] = user_id
            session['login_message_shown'] = False
            session['auth_method'] = 'google'
            
            # CRITICAL FIX: Ensure session is marked as modified
            session.modified = True
            session.permanent = False
            
            print(f"✅ Session created for user: {user_id}")
            
        except Exception as session_error:
            print(f"❌ Session creation error: {session_error}")
            return jsonify({'success': False, 'message': 'Session error'}), 500

        flash("Logged In Successfully", "success")

        response_data = {
            'success': True, 
            'message': 'Google login successful',
            'redirect_url': url_for('dashboard'),
            'user_id': user_id
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Unexpected error in google_login: {e}")
        return jsonify({
            'success': False, 
            'message': f'Authentication failed: {str(e)}'
        }), 500
        

# -------------------- REGISTER ROUTES --------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if email != session.get('verified_email'):
            flash("Please verify your email first", "danger")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters long!", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        username_query = db.collection('user').where('username', '==', username).limit(1).get()
        if len(username_query) > 0:
            flash("Username already taken!", "danger")
            return redirect(url_for('register'))

        email_query = db.collection('user').where('email', '==', email).limit(1).get()
        if len(email_query) > 0:
            flash("Email already taken!", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        
        new_user = {
            'name': name,
            'username': username,
            'email': email,
            'password': hashed_password,
            'auth_method': 'manual',
            'datetime_login': None,
            'datetime_logout': None
        }
        
        try:
            db.collection('user').add(new_user)
            session.pop('verified_email', None)
            flash("Registration successful! You can now login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash("Registration failed. Please try again.", "danger")
            print(f"Registration error: {e}")
        
    return render_template('register.html')


# -------------------- USERNAME CHECK ROUTE --------------------

@app.route('/check_username', methods=['POST'])
def check_username():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        
        if not username:
            return jsonify({'available': False, 'message': 'Username is required'})
        
        if len(username) < 3:
            return jsonify({'available': False, 'message': 'Username must be at least 3 characters'})
        
        import re
        if not re.match("^[a-zA-Z0-9_.-]+$", username):
            return jsonify({'available': False, 'message': 'Username can only contain letters, numbers, dots, hyphens, and underscores'})
        
        username_query = db.collection('user').where('username', '==', username).limit(1).get()
        
        if len(username_query) > 0:
            return jsonify({'available': False, 'message': 'Username already exists'})
        else:
            return jsonify({'available': True, 'message': 'Username is available'})
            
    except Exception as e:
        print(f"Error checking username: {str(e)}")
        return jsonify({'available': False, 'message': 'Error checking username'}), 500


# -------------------- EMAIL CHECK ROUTE --------------------

@app.route('/check_email', methods=['POST'])
def check_email():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'available': False, 'message': 'Email is required'})
        
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(email):
            return jsonify({'available': False, 'message': 'Please enter a valid email address'})
        
        email_query = db.collection('user').where('email', '==', email).limit(1).get()
        
        if len(email_query) > 0:
            return jsonify({'available': False, 'message': 'Email is already registered'})
        else:
            return jsonify({'available': True, 'message': 'Email is available'})
            
    except Exception as e:
        print(f"Error checking email: {str(e)}")
        return jsonify({'available': False, 'message': 'Error checking email'}), 500


# -------------------- FORGOT/RESET PASSWORD ROUTES --------------------

reset_codes = {}

def send_reset_email(email, reset_code):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv('SMTP_EMAIL', 'your_email@gmail.com')
        sender_password = os.getenv('SMTP_PASSWORD', 'your_app_password')
        
        message = MIMEMultipart("alternative")
        message["Subject"] = "MANGOSQUEEN Password Reset"
        message["From"] = sender_email
        message["To"] = email
        
        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #6b46c1;">MANGOSQUEEN Password Reset</h2>
                <p>You requested to reset your password. Use the verification code below:</p>
                <div style="background-color: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; text-align: center; margin: 20px 0;">
                    <h3 style="margin: 0; color: #6b46c1; font-size: 24px; letter-spacing: 5px;">{reset_code}</h3>
                </div>
                <p>This code will expire in 1 minute for security reasons.</p>
                <p>If you didn't request this reset, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
                <p style="color: #718096; font-size: 14px;">MANGOSQUEEN Team</p>
            </div>
        </body>
        </html>
        """
        
        message.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
            
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

@app.route('/forgot-password', methods=['POST'])
def forgot_password_ajax():
    email = request.form['email']
    
    users_ref = db.collection('user')
    query = users_ref.where('email', '==', email).limit(1)
    users = query.get()
    
    if len(users) == 0:
        return jsonify({'success': False, 'message': 'Email does not exists.'})
    
    reset_code = str(random.randint(100000, 999999))
    
    reset_codes[email] = {
        'code': reset_code,
        'expires': datetime.now().timestamp() + 600  # 10 minutes
    }
    
    if send_reset_email(email, reset_code):
        return jsonify({'success': True, 'message': 'Verification code sent to your email.'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send email. Please try again.'})

@app.route('/verify-reset-code', methods=['POST'])
def verify_reset_code_ajax():
    email = request.form['email']
    entered_code = request.form['code']
    
    if email in reset_codes:
        stored_code = reset_codes[email]
        
        if datetime.now().timestamp() > stored_code['expires']:
            return jsonify({'success': False, 'message': 'Verification code has expired. Please request a new one.'})
        
        if entered_code == stored_code['code']:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid verification code.'})
    else:
        return jsonify({'success': False, 'message': 'Verification code not found or expired. Please request a new one.'})

@app.route('/reset-password', methods=['POST'])
def reset_password_ajax():
    email = request.form['email']
    password = request.form['password']
    
    users_ref = db.collection('user')
    query = users_ref.where('email', '==', email).limit(1)
    users = query.get()
    
    if len(users) == 1:
        user_doc = users[0]
        hashed_password = generate_password_hash(password)
        
        users_ref.document(user_doc.id).update({
            'password': hashed_password
        })
        
        if email in reset_codes:
            del reset_codes[email]
        
        return jsonify({'success': True, 'message': 'Password reset successfully!'})
    else:
        return jsonify({'success': False, 'message': 'User not found. Please try again.'})
    
    
# -------------------- REGISTER EMAIL VERIFICATION --------------------

verification_codes = {}

def send_verification_email(email, verification_code):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv('SMTP_EMAIL', 'your_email@gmail.com')
        sender_password = os.getenv('SMTP_PASSWORD', 'your_app_password')
        
        message = MIMEMultipart("alternative")
        message["Subject"] = "MANGOSQUEEN Email Verification"
        message["From"] = sender_email
        message["To"] = email
        
        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #6b46c1;">MANGOSQUEEN Email Verification</h2>
                <p>Thank you for registering with MANGOSQUEEN! Please use the verification code below to verify your email:</p>
                <div style="background-color: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; text-align: center; margin: 20px 0;">
                    <h3 style="margin: 0; color: #6b46c1; font-size: 24px; letter-spacing: 5px;">{verification_code}</h3>
                </div>
                <p>This code will expire in 1 minute for security reasons.</p>
                <p>If you didn't request this verification, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
                <p style="color: #718096; font-size: 14px;">MANGOSQUEEN Team</p>
            </div>
        </body>
        </html>
        """
        
        message.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
            
        return True
    except Exception as e:
        print(f"Error sending verification email: {e}")
        return False

@app.route('/send_verification', methods=['POST'])
def send_verification():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'success': False, 'message': 'Email is required'})
    
    email_query = db.collection('user').where('email', '==', email).limit(1).get()
    if len(email_query) > 0:
        return jsonify({'success': False, 'message': 'Email already registered'})
    
    code = ''.join(random.choices(string.digits, k=6))
    
    verification_codes[email] = {
        'code': code,
        'expires': datetime.now() + timedelta(minutes=1)
    }
    
    if send_verification_email(email, code):
        return jsonify({'success': True, 'message': 'Verification code sent to your email'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification email. Please try again.'})

@app.route('/verify_code', methods=['POST'])
def verify_code():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    
    if not email or not code:
        return jsonify({'success': False, 'message': 'Email and code are required'})
    
    if email not in verification_codes:
        return jsonify({'success': False, 'message': 'No verification code found for this email'})
    
    if datetime.now() > verification_codes[email]['expires']:
        del verification_codes[email]
        return jsonify({'success': False, 'message': 'Verification code has expired'})
    
    if code == verification_codes[email]['code']:
        session['verified_email'] = email
        del verification_codes[email]
        return jsonify({'success': True, 'message': 'Email verified successfully'})
    else:
        return jsonify({'success': False, 'message': 'Invalid verification code'})


# -------------------- ML MODEL SETUP (UPDATED FOR TFLITE) --------------------
# Define the base directory (where app.py is located)
basedir = os.path.dirname(os.path.abspath(__file__))

# --- UPDATED: Path to the new TFLite model ---
MODEL_PATH = os.path.join(basedir, 'static', 'models', '1mangosqueen.tflite')

# --- UPDATED: Load TFLite Model ---
interpreter = None
input_details = None
output_details = None

try:
    print(f"⏳ Loading TFLite model from: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ TFLite Model loaded successfully.")
    print(f"   Input Shape: {input_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

DISEASE_CLASSES = ['Anthracnose', 'Dry Rot', 'Gummosis', 'Half ripe', 'Healthy ripe', 'Over ripe', 'Raw', 'Unknown']

PREVENTIVE_METHODS = {
    'Anthracnose': [
        "Apply fungicides during flowering and early fruit development",
        "Prune trees to improve air circulation",
        "Collect and destroy fallen infected fruits",
        "Maintain proper tree spacing"
    ],
    'Dry Rot': [
        "Avoid fruit injuries during harvesting and handling",
        "Ensure proper drying and storage conditions",
        "Dispose of infected fruits immediately",
        "Apply post-harvest treatments if necessary"
    ],
    'Gummosis': [
        "Remove infected bark and apply wound dressing",
        "Improve soil drainage to prevent root stress",
        "Avoid trunk injuries from tools or machinery",
        "Apply copper-based fungicides as needed"
    ],
    'Half ripe': [
        "Harvest only when the fruit is fully mature",
        "Monitor for early ripening signs to avoid premature harvest",
        "Ensure proper handling to prevent bruising"
    ],
    'Healthy ripe': [
        "Continue regular watering and fertilization",
        "Ensure hygienic handling and storage",
        "Harvest at the right time for optimal quality"
    ],
    'Over ripe': [
        "Harvest fruits at proper maturity to avoid overripening",
        "Inspect frequently during peak ripening periods",
        "Use overripe fruits quickly or process into value-added products"
    ],
    'Raw': [
        "Allow more time for the fruit to ripen naturally",
        "Avoid early harvesting to maintain quality",
        "Store in optimal conditions for ripening"
    ],
    'Unknown': [
        "Unable to determine fruit condition",
        "Please retake or upload a clearer image",
        "Consult with an agricultural expert if issues persist"
    ]
}

# -------------------- HELPER FUNCTIONS --------------------

def get_logged_in_status():
    return session.get('logged_in', False)

def get_user_id():
    return session.get('user_id')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- UPDATED: Helper for Standard Preprocessing ---
def preprocess_image_for_model(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# --- UPDATED: Predict function using TFLite ---
def predict_disease(image_path):
    try:
        if interpreter is None:
            return {
                'disease_name': "Model not loaded",
                'confidence_score': 0.0,
                'preventive_methods': ["Please contact system administrator"]
            }
            
        img = Image.open(image_path)
        img_array = preprocess_image_for_model(img)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predictions = output_data[0]
        predicted_class_index = np.argmax(predictions)
        confidence_score = float(predictions[predicted_class_index]) * 100
        
        disease_name = DISEASE_CLASSES[predicted_class_index]
        preventive_methods = PREVENTIVE_METHODS[disease_name]
        
        return {
            'disease_name': disease_name,
            'confidence_score': confidence_score,
            'preventive_methods': preventive_methods
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            'disease_name': "Error in prediction",
            'confidence_score': 0.0,
            'preventive_methods': ["Please try again with a clearer image"]
        }

# --- UPDATED: Real-time prediction using TFLite ---
def predict_disease_from_base64(image_data):
    try:
        if interpreter is None:
            return {
                'predicted_class': "Model not loaded",
                'confidence': 0.0,
                'preventive_methods': ["Please contact system administrator"]
            }
        
        if image_data.startswith('data:image/'):
            image_format, base64_data = image_data.split(';base64,')
        else:
            base64_data = image_data
        
        image_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(image_bytes))
        img_array = preprocess_image_for_model(img)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predictions = output_data[0]
        predicted_class_index = np.argmax(predictions)
        confidence_score = float(predictions[predicted_class_index]) * 100
        
        disease_name = DISEASE_CLASSES[predicted_class_index]
        preventive_methods = PREVENTIVE_METHODS[disease_name]
        
        return {
            'predicted_class': disease_name,
            'confidence': confidence_score,
            'preventive_methods': preventive_methods
        }
    except Exception as e:
        print(f"Error in real-time prediction: {e}")
        return {
            'predicted_class': "Error in prediction",
            'confidence': 0.0,
            'preventive_methods': ["Please try again with a clearer image"]
        }

def save_scan_to_db(scan_data):
    try:
        doc_ref = db.collection('diseasedetection').add(scan_data)
        return doc_ref[1].id
    except Exception as e:
        print(f"Database error: {e}")
        raise e

def process_image_upload(file, user_id):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    prediction_results = predict_disease(file_path)
    
    return {
        'user_id': user_id,
        'image': os.path.join('uploads', filename).replace('\\', '/'),
        'prediction': prediction_results['disease_name'],
        'confidence': prediction_results['confidence_score'],
        'datetime_scanned': datetime.now().isoformat()
    }, prediction_results

def process_camera_capture(image_data, user_id):
    if image_data.startswith('data:image/'):
        image_format, base64_data = image_data.split(';base64,')
        image_ext = image_format.split('/')[-1]
    else:
        base64_data = image_data
        image_ext = 'jpg'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"camera_capture_{timestamp}.{image_ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))
    
    prediction_results = predict_disease(file_path)
    
    return {
        'user_id': user_id,
        'image': os.path.join('uploads', filename).replace('\\', '/'),
        'prediction': prediction_results['disease_name'],
        'confidence': prediction_results['confidence_score'],
        'datetime_scanned': datetime.now().isoformat()
    }, prediction_results

# -------------------- SCAN & PROCESSING ROUTES --------------------
@app.route('/scan', methods=['GET', 'POST'])
@login_required 
def scan():
    if request.method == 'POST':
        if 'image_data' in request.form and request.form['image_data']:
            try:
                image_data = request.form['image_data']
                scan_data, prediction_results = process_camera_capture(image_data, session['user_id'])
                
                try:
                    scan_id = save_scan_to_db(scan_data)
                    scan_data['id'] = scan_id
                except Exception as e:
                    flash(f"Database error: {str(e)}", "danger")
                    return redirect(url_for('scan'))
                
                session['image_path'] = scan_data['image']
                session['disease_name'] = scan_data['prediction']
                session['confidence_score'] = scan_data['confidence']
                session['preventive_methods'] = prediction_results['preventive_methods']
                
                return redirect(url_for('results'))
            except Exception as e:
                flash(f"Error processing camera image: {str(e)}", "danger")
                print(f"Error processing camera image: {str(e)}")
                
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash("No file selected", "danger")
                return redirect(request.url)
                
            if file and allowed_file(file.filename):
                try:
                    scan_data, prediction_results = process_image_upload(file, session['user_id'])
                    
                    try:
                        scan_id = save_scan_to_db(scan_data)
                        scan_data['id'] = scan_id
                    except Exception as e:
                        flash(f"Database error: {str(e)}", "danger")
                        return redirect(url_for('scan'))
                    
                    session['image_path'] = scan_data['image']
                    session['disease_name'] = scan_data['prediction']
                    session['confidence_score'] = scan_data['confidence']
                    session['preventive_methods'] = prediction_results['preventive_methods']
                    
                    return redirect(url_for('results'))
                except Exception as e:
                    flash(f"Error processing image: {str(e)}", "danger")
            else:
                flash("File type not allowed. Please upload JPG, JPEG or PNG images only", "danger")
        else:
            flash("No image data received", "danger")
            
    return render_template('scan.html', logged_in=True)

@app.route('/results')
@login_required 
def results():
    image_path = session.get('image_path')
    disease_name = session.get('disease_name')
    confidence_score = session.get('confidence_score')
    preventive_methods = session.get('preventive_methods')
    
    if not all([image_path, disease_name is not None, confidence_score is not None, preventive_methods]):
        flash("No scan results found. Please scan an image first.", "warning")
        return redirect(url_for('scan'))
    
    return render_template('results.html', 
                           logged_in=True,
                           image_path=image_path,
                           disease_name=disease_name, 
                           confidence_score=confidence_score,
                           preventive_methods=preventive_methods)

# -------------------- REAL-TIME DETECTION ROUTES --------------------
@app.route('/realtime-scan', methods=['POST'])
@login_required 
def realtime_scan():
    try:
        if 'image_data' not in request.form:
            return jsonify({'success': False, 'error': 'No image data received'}), 400
        
        image_data = request.form['image_data']
        prediction_results = predict_disease_from_base64(image_data)
        
        return jsonify({
            'success': True,
            'predicted_class': prediction_results['predicted_class'],
            'confidence': prediction_results['confidence'],
            'preventive_methods': prediction_results['preventive_methods']
        })
        
    except Exception as e:
        print(f"Error in real-time scan: {e}")
        return jsonify({
            'success': False,
            'error': f'Error processing frame: {str(e)}'
        }), 500

@app.route('/save-detection', methods=['POST'])
@login_required 
def save_detection():
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Session expired'}), 401
        
        if 'image_data' not in request.form or 'detection_data' not in request.form:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
        
        image_data = request.form['image_data']
        detection_info = json.loads(request.form['detection_data'])
        
        if image_data.startswith('data:image/'):
            image_format, base64_data = image_data.split(';base64,')
            image_ext = image_format.split('/')[-1]
        else:
            base64_data = image_data
            image_ext = 'jpg'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"realtime_detection_{timestamp}.{image_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        
        scan_data = {
            'user_id': session['user_id'],
            'image': os.path.join('uploads', filename).replace('\\', '/'),
            'prediction': detection_info['predicted_class'],
            'confidence': float(detection_info['confidence']),
            'datetime_scanned': datetime.now().isoformat()
        }
        
        try:
            scan_id = save_scan_to_db(scan_data)
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Database error: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Real-time detection saved successfully',
            'scan_id': scan_id
        })
        
    except Exception as e:
        print(f"Error saving detection: {e}")
        return jsonify({
            'success': False,
            'error': f'Error saving detection: {str(e)}'
        }), 500

# -------------------- PROFILE & SETTINGS ROUTES --------------------
@app.route('/sort')
@login_required 
def sort():
    scans = []
    try:
        # 1. Kunin ang mga folders ng petsa sa 'sorts' collection
        # Kukuha tayo ng documents kung saan match ang user_id
        date_docs = db.collection('sorts').where('user_id', '==', session['user_id']).get()

        # 2. Isa-isahin ang bawat petsa para kunin ang laman (records)
        for day_doc in date_docs:
            # Pumasok sa loob ng sub-collection na 'records'
            records = day_doc.reference.collection('records').order_by('datetime_scanned', direction=firestore.Query.DESCENDING).get()
            
            for rec in records:
                scan_data = rec.to_dict()
                scan_data['id'] = rec.id
                
                # Siguraduhin natin na tama ang format para sa HTML
                # (Kung 'prediction' ang gamit sa HTML, dapat meron nito)
                if 'prediction' not in scan_data:
                    scan_data['prediction'] = 'Unknown'
                    
                scans.append(scan_data)

        # 3. Ayusin ang final list mula sa pinakabago (Newest to Oldest)
        # Dahil galing sa iba't ibang folder, manual sort natin sa Python side
        scans.sort(key=lambda x: x.get('datetime_scanned', ''), reverse=True)

    except Exception as e:
        print(f"Error fetching sort history: {e}")
        scans = []
    
    return render_template('sort.html', logged_in=True, scans=scans)



@app.route('/about')
@login_required 
def about():
    return render_template('about.html', logged_in=True)

# --- REFACTORED PROFILE LOGIC ---

def send_verification_email_for_profile(email, code):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv('SMTP_EMAIL', 'your_email@gmail.com')
        sender_password = os.getenv('SMTP_PASSWORD', 'your_app_password')
        
        if sender_email == 'your_email@gmail.com' or sender_password == 'your_app_password':
            print("Email credentials not configured properly")
            return False
        
        message = MIMEMultipart("alternative")
        message["Subject"] = "MANGOSQUEEN Profile Update Verification"
        message["From"] = sender_email
        message["To"] = email
        
        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #6b46c1;">MANGOSQUEEN Profile Update Verification</h2>
                <p>You requested to update your profile information. Please use the verification code below:</p>
                <div style="background-color: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; text-align: center; margin: 20px 0;">
                    <h3 style="margin: 0; color: #6b46c1; font-size: 24px; letter-spacing: 5px;">{code}</h3>
                </div>
                <p>This code will expire in 10 minutes for security reasons.</p>
                <p>If you didn't request this update, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
                <p style="color: #718096; font-size: 14px;">MANGOSQUEEN Team</p>
            </div>
        </body>
        </html>
        """
        
        message.attach(MIMEText(html, "html"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
            
        print(f"Verification email sent successfully to {email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"Error sending profile verification email: {e}")
        return False

@app.route('/send-profile-verification', methods=['POST'])
@login_required
def send_profile_verification():
    if not session.get('email'):
        return jsonify({'success': False, 'message': 'User email not found.'})
    
    email = session['email']
    code = ''.join(random.choices(string.digits, k=6))
    
    session['profile_verification_code'] = code
    session['profile_verification_expires'] = (datetime.now() + timedelta(minutes=10)).isoformat()
    
    if send_verification_email_for_profile(email, code):
        return jsonify({'success': True, 'message': 'Verification code sent successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send verification code'})

@app.route('/profile', methods=['GET'])
@login_required 
def profile():
    user = None
    
    if 'user_id' in session:
        try:
            user_doc = db.collection('user').document(session['user_id']).get()
            if user_doc.exists:
                user = user_doc.to_dict()
                user['id'] = user_doc.id
                if 'auth_method' not in user:
                    user['auth_method'] = 'manual'
        except Exception as e:
            print(f"Error fetching user by ID: {e}")
    
    if user is None and 'username' in session:
        try:
            username = session.get('username')
            users = db.collection('user').where('username', '==', username).limit(1).get()
            if users:
                user_doc = users[0]
                user = user_doc.to_dict()
                user['id'] = user_doc.id
        except Exception as e:
            print(f"Error fetching user by username: {e}")

    if user is None:
        flash("User session expired. Please log in again.", "warning")
        session.clear()
        return redirect(url_for('login'))
    
    return render_template('profile.html', logged_in=True, user=user)

@app.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    try:
        user_id = session.get('user_id')
        user_doc_ref = db.collection('user').document(user_id)
        user_doc = user_doc_ref.get()
        
        if not user_doc.exists:
            return jsonify({'success': False, 'message': 'User not found.'})
            
        user_data = user_doc.to_dict()
        auth_method = user_data.get('auth_method', 'manual')
        
        data = request.get_json()
        new_name = data.get('name', '').strip()
        new_username = data.get('username', '').strip()
        
        if not new_name or not new_username:
            return jsonify({'success': False, 'message': 'Name and Username are required.'})
            
        if new_username != user_data.get('username'):
            check = db.collection('user').where('username', '==', new_username).limit(1).get()
            if len(check) > 0:
                 return jsonify({'success': False, 'message': 'Username already taken.'})

        update_payload = {
            'name': new_name,
            'username': new_username
        }
        
        if auth_method == 'manual':
            current_password = data.get('current_password', '')
            if not current_password:
                return jsonify({'success': False, 'message': 'Current password is required.'})
            
            if not user_data.get('password'):
                 return jsonify({'success': False, 'message': 'Account error. Contact support.'})

            if not check_password_hash(user_data.get('password', ''), current_password):
                return jsonify({'success': False, 'message': 'Incorrect current password.'})
            
            if data.get('change_password'):
                new_pass = data.get('new_password', '')
                if len(new_pass) < 6:
                    return jsonify({'success': False, 'message': 'New password must be at least 6 characters.'})
                update_payload['password'] = generate_password_hash(new_pass)
                
        elif auth_method == 'google':
            entered_code = data.get('verification_code', '')
            stored_code = session.get('profile_verification_code')
            expiry = session.get('profile_verification_expires')
            
            if not entered_code:
                return jsonify({'success': False, 'message': 'Verification code required.', 'field': 'code'})
            
            if not stored_code or not expiry:
                return jsonify({'success': False, 'message': 'Please request a new code.', 'field': 'code'})
                
            if datetime.now() > datetime.fromisoformat(expiry):
                session.pop('profile_verification_code', None)
                return jsonify({'success': False, 'message': 'Verification code has expired.', 'field': 'code'})
                
            if entered_code != stored_code:
                return jsonify({'success': False, 'message': 'Invalid verification code.', 'field': 'code'})
                
            session.pop('profile_verification_code', None)

        user_doc_ref.update(update_payload)
        session['username'] = new_username
        
        return jsonify({'success': True, 'message': 'Save. You successfully update your profile.'})

    except Exception as e:
        print(f"Update error: {e}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

# -------------------- API ENDPOINTS --------------------

@app.route('/api/disease-data')
@login_required 
def get_disease_data():
    disease_filter = request.args.get('diseaseType', 'all')
    date_from = request.args.get('dateFrom')
    date_to = request.args.get('dateTo')

    try:
        query = db.collection('diseasedetection').where('user_id', '==', session['user_id'])

        if disease_filter != 'all':
            query = query.where('prediction', '==', disease_filter)
        
        if date_from and date_to:
            query = query.where('datetime_scanned', '>=', date_from).where('datetime_scanned', '<=', date_to)

        docs = query.get()
        scans = []
        for doc in docs:
            scan_data = doc.to_dict()
            scan_data['id'] = doc.id
            scans.append(scan_data)

        disease_types = ['Anthracnose', 'Dry Rot', 'Gummosis', 'Half ripe', 'Healthy ripe', 'Over ripe', 'Raw']
        scan_counts = [0] * len(disease_types)
        
        for scan in scans:
            if scan['prediction'] in disease_types:
                index = disease_types.index(scan['prediction'])
                scan_counts[index] += 1

        return jsonify({
            "diseaseTypes": disease_types,
            "scanCounts": scan_counts,
            "history": [{
                "id": scan['id'],
                "dateTime": scan['datetime_scanned'],
                "filename": scan['image'].split('/')[-1],
                "diseaseType": scan['prediction'],
                "confidence": round(scan['confidence'], 2)
            } for scan in scans]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete-scan/<scan_id>', methods=['DELETE'])
@login_required 
def delete_scan(scan_id):
    try:
        doc = db.collection('diseasedetection').document(scan_id).get()
        if not doc.exists:
            return jsonify({"error": "Scan not found"}), 404
            
        scan_data = doc.to_dict()
        if scan_data['user_id'] != session['user_id']:
            return jsonify({"error": "Access denied"}), 403
        
        db.collection('diseasedetection').document(scan_id).delete()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- ROBUST CHATBOT ROUTE (FIXED) --------------------
@app.route('/api/chat', methods=['POST'])
@login_required 
def chat():
    if not model:
        print("❌ ERROR: Gemini Model is not configured. Check your API Key.")
        return jsonify({'reply': "System Error: API Key is missing or invalid."})

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        print(f"📩 Received message: {user_message}") 

        chat_history = session.get('chat_history', [])
        
        chat_session = model.start_chat(history=chat_history)
        
        response = chat_session.send_message(user_message)
        
        try:
            bot_reply = response.text
        except ValueError:
            print("⚠️ WARNING: Gemini blocked the response due to safety filters.")
            bot_reply = "I apologize, but I cannot answer that query due to safety guidelines."

        print(f"🤖 Bot Reply: {bot_reply}") 

        chat_history.append({'role': 'user', 'parts': [user_message]})
        chat_history.append({'role': 'model', 'parts': [bot_reply]})
        session['chat_history'] = chat_history[-20:] 
        session.modified = True

        return jsonify({'reply': bot_reply})

    except Exception as e:
        print(f"❌ CRITICAL ERROR in /api/chat: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({'reply': f"Error: {str(e)}"}), 500
    
# -------------------- ERROR HANDLERS --------------------

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Resource not found'}), 404
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Internal server error'}), 500
    return render_template('500.html'), 500


# -----------------------------------------------------------------------



# ==========================================
#  HARDWARE BRIDGE & ANALYTICS (NEW)
# ==========================================

# Raspi connection config
RASPI_HOST = os.getenv('RASPI_HOST', '192.168.137.191')  # IP of your Machine
RASPI_VIDEO_PATH = os.getenv('RASPI_VIDEO_PATH', f'http://{RASPI_HOST}:5000/video_feed')
RASPI_SERVO_URL = os.getenv('RASPI_SERVO_URL', f'http://{RASPI_HOST}:5000/move_servo')

# 1. Check if Machine is Online
@app.route('/api/raspi_ping')
@login_required
def raspi_ping():
    try:
        r = requests.get(RASPI_VIDEO_PATH, stream=True, timeout=5)
        status = r.status_code
        ctype = r.headers.get('Content-Type', '')
        try:
            chunk = next(r.iter_content(chunk_size=1024))
        except Exception:
            chunk = b''
        r.close()
        ok = (status == 200) and ('multipart' in ctype.lower() or chunk)
        return jsonify({'ok': ok, 'status': status, 'content_type': ctype})
    except Exception as e:
        app.logger.warning(f"raspi_ping error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 503

# 2. Proxy Video Stream (Fixes HTTPS/CORS issues)
@app.route('/raspi_stream')
@login_required
def raspi_stream():
    def stream_generator():
        try:
            with requests.get(RASPI_VIDEO_PATH, stream=True, timeout=(3, 10)) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
        except Exception as e:
            app.logger.warning(f"Error proxying raspi stream: {e}")
            return
    try:
        head = requests.head(RASPI_VIDEO_PATH, timeout=3)
        content_type = head.headers.get('Content-Type', 'multipart/x-mixed-replace;boundary=frame')
    except Exception:
        content_type = 'multipart/x-mixed-replace; boundary=frame'
    return Response(stream_generator(), mimetype=content_type)

# 3. Control Servo Motor
@app.route('/api/servo', methods=['POST'])
@login_required
def api_servo():
    data = request.get_json(force=True, silent=True) or {}
    direction = data.get('direction')
    if direction not in ('left', 'right', 'center'):
        return jsonify({'error': 'Invalid direction'}), 400
    try:
        resp = requests.post(RASPI_SERVO_URL, json={'direction': direction}, timeout=4)
        return jsonify({'success': True, 'raspi_status': resp.status_code}), 200
    except Exception as e:
        print(f"Servo proxy error: {e}")
        return jsonify({'error': str(e)}), 500

# 4. Generate Performance Graphs
@app.route('/api/sort-analytics')
@login_required
def sort_analytics():
    try:
        period = request.args.get('period', 'day').lower()
        now = datetime.now()
        labels = []
        buckets = {}

        # Set Time Range
        if period == 'week':
            weeks = 12
            for i in range(weeks-1, -1, -1):
                d = now - timedelta(weeks=i)
                y, w, _ = d.isocalendar()
                label = f"{y}-W{w:02d}"
                labels.append(label)
                buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}
        elif period == 'month':
            months = 12
            for i in range(months-1, -1, -1):
                month_date = (now.replace(day=1) - timedelta(days=30*i))
                year = month_date.year
                month = month_date.month
                label = f"{year}-{month:02d}"
                labels.append(label)
                buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}
        else: # Default: Day
            days = 30
            for i in range(days-1, -1, -1):
                d = now - timedelta(days=i)
                label = d.strftime('%Y-%m-%d')
                labels.append(label)
                buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}

        # Fetch Data
        sorts_q = db.collection('sorts').where('user_id', '==', session['user_id']).get()
        history = []

        for day_doc in sorts_q:
            day = day_doc.to_dict()
            date_key = day.get('date')
            if not date_key: continue

            try:
                dt_obj = datetime.fromisoformat(date_key + 'T00:00:00')
            except Exception:
                try: dt_obj = datetime.strptime(date_key, '%Y-%m-%d')
                except Exception: continue

            if period == 'week':
                y, w, _ = dt_obj.isocalendar()
                label = f"{y}-W{w:02d}"
            elif period == 'month':
                label = dt_obj.strftime('%Y-%m')
            else:
                label = dt_obj.strftime('%Y-%m-%d')

            accepted_count = int(day.get('accepted') or 0)
            rejected_count = int(day.get('rejected') or 0)
            total_count = int(day.get('total') or (accepted_count + rejected_count))

            if label in buckets:
                buckets[label]['accepted'] += accepted_count
                buckets[label]['rejected'] += rejected_count
                buckets[label]['total'] += total_count

        accepted = [buckets[l]['accepted'] for l in labels]
        rejected = [buckets[l]['rejected'] for l in labels]
        total = [buckets[l]['total'] for l in labels]

        return jsonify({
            'labels': labels,
            'accepted': accepted,
            'rejected': rejected,
            'total': total,
            'history': history
        })
    except Exception as e:
        print(f"Error in sort_analytics: {e}")
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)