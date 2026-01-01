from flask import Flask, render_template, redirect, url_for, session, request, flash, jsonify, make_response, Response
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
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image  # already imported in file, kept for clarity
import firebase_admin
from firebase_admin import credentials, firestore, auth
import tensorflow as tf
from functools import wraps

# Define base directory and load .env
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = Flask(__name__)

# ==========================================
#  GEMINI AI & CHATBOT CONFIGURATION (V2 UPGRADE)
# ==========================================

# 1. Load API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# 2. Define System Instructions (The "Brain") - Uses V2 Permissive Logic
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
3. GENERAL TOPICS: 
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

# 3. Initialize Gemini Model - Uses V2 (Gemini 2.0 Flash)
model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Configure Model with System Instructions
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

# SESSION CONFIGURATION (V2 SECURE)
app.config.update(
    SESSION_COOKIE_SECURE=True,           
    SESSION_COOKIE_HTTPONLY=True,         
    SESSION_COOKIE_SAMESITE='Lax',        
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),  
    SESSION_PERMANENT=False,              # Non-persistent sessions (Secure)
)

from werkzeug.middleware.proxy_fix import ProxyFix
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

# -------------------- SESSION SECURITY DECORATOR (V2) --------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Check if user is logged in
        if 'logged_in' not in session or not session['logged_in']:
            flash("Please log in to access the page.", "warning")
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
@login_required 
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
        
        # 1. Try to find user by EMAIL
        query = users_ref.where('email', '==', login_input.lower()).limit(1)
        users = query.get()
        
        # 2. If not found by email, try to find by USERNAME
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
                
                # Securely store email from DB
                session['email'] = user['email']
                
                session['user_id'] = user['id']
                session['login_message_shown'] = False
                session.permanent = False  # V2 Fix: Ensure non-persistent
                
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
    
    session.clear() # V2 Fix: Completely clear session
    
    flash("Logged Out Successfully", "info")
    return redirect(url_for('home'))


# -------------------- CORS HELPER FUNCTIONS --------------------

def add_cors_headers(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        origin = request.headers.get('Origin', '')
        allowed_origins = [
            'https://mangosqueen.onrender.com', 
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
        
        if request.method == 'OPTIONS':
            response = jsonify({'success': True})
            response.headers.extend(response_headers)
            return response
        
        response = f(*args, **kwargs)
        
        if isinstance(response, tuple):
            if len(response) == 3:
                data, status, headers = response
                if headers is None:
                    headers = {}
                headers.update(response_headers)
                return data, status, headers
            elif len(response) == 2:
                data, status = response
                return data, status, response_headers
            else:
                return response[0], 200, response_headers
        else:
            return response, 200, response_headers
    
    return decorated_function


# -------------------- CONTINUE WITH EMAIL ROUTES (V2 ROBUST) --------------------

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
            # V2 Fix: Added clock_skew_seconds for reliability
            decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=10)
            uid = decoded_token['uid']
            print(f"✅ Token verified for UID: {uid}, Email: {decoded_token.get('email')}")
            
            token_email = decoded_token.get('email')
            if token_email and email and token_email.lower() != email.lower():
                print(f"❌ Email mismatch: token={token_email}, provided={email}")
                return jsonify({'success': False, 'message': 'Email verification failed'}), 401
                
        except ValueError as ve:
            print(f"❌ Token validation error (ValueError): {ve}")
            return jsonify({'success': False, 'message': 'Invalid token format'}), 401
        except auth.ExpiredIdTokenError:
            print("❌ Token has expired")
            return jsonify({'success': False, 'message': 'Token has expired'}), 401
        except auth.RevokedIdTokenError:
            print("❌ Token has been revoked")
            return jsonify({'success': False, 'message': 'Token has been revoked'}), 401
        except auth.InvalidIdTokenError:
            print("❌ Invalid token")
            return jsonify({'success': False, 'message': 'Invalid token'}), 401
        except Exception as verify_error:
            print(f"❌ Token verification error: {verify_error}")
            return jsonify({'success': False, 'message': 'Invalid ID token'}), 401

        users_ref = db.collection('user')
        query = users_ref.where('email', '==', email).limit(1)
        users = query.get()

        user_id = None
        
        try:
            if len(users) > 0:
                user_doc = users[0]
                user_id = user_doc.id
                print(f"🔄 Updating existing user: {user_id}")
                
                update_data = {
                    'datetime_login': datetime.now().isoformat(),
                    'auth_method': 'google'
                }
                
                if photo_url:
                    existing_user = user_doc.to_dict()
                    if existing_user.get('photo_url') != photo_url:
                        update_data['photo_url'] = photo_url
                
                users_ref.document(user_id).update(update_data)
                print(f"✅ Updated existing user: {user_id}")
                
            else:
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
                print(f"✅ Created new user: {user_id}")
                
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': 'Database error'}), 500

        if not user_id:
            print("❌ No user ID obtained from database operations")
            return jsonify({'success': False, 'message': 'User creation failed'}), 500

        try:
            session.clear() # V2 Fix: Clean state
            
            session['logged_in'] = True
            session['username'] = email.split('@')[0]
            session['email'] = email
            session['user_id'] = user_id
            session['login_message_shown'] = False
            session['auth_method'] = 'google'
            
            session.permanent = False  # V2 Fix: Non-persistent
            
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

        print(f"🎉 Google login successful for user: {user_id}")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Unexpected error in google_login: {e}")
        import traceback
        traceback.print_exc()
        
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


# -------------------- ML MODEL SETUP (V2 TFLITE) --------------------
# Define the base directory (where app.py is located)
basedir = os.path.dirname(os.path.abspath(__file__))

# Path to the new TFLite model
MODEL_PATH = os.path.join(basedir, 'static', 'models', '1mangosqueen.tflite')

# Load TFLite Model
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

# Standard Preprocessing
def preprocess_image_for_model(img):
    """
    Standard preprocessing for MobileNetV2 trained models.
    Scales values to [-1, 1].
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to float32 array
    img_array = np.array(img, dtype=np.float32)
    
    # Expand dimensions (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale to [-1, 1] instead of [0, 1]
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# Predict function using TFLite
def predict_disease(image_path):
    """Predict disease from image file path using TFLite"""
    try:
        if interpreter is None:
            return {
                'disease_name': "Model not loaded",
                'confidence_score': 0.0,
                'preventive_methods': ["Please contact system administrator"]
            }
            
        # Load image with PIL
        img = Image.open(image_path)
        
        # Preprocess
        img_array = preprocess_image_for_model(img)
        
        # TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get result from the first batch
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

# Real-time prediction using TFLite
def predict_disease_from_base64(image_data):
    """Predict disease from base64 image data - used for real-time detection"""
    try:
        if interpreter is None:
            return {
                'predicted_class': "Model not loaded",
                'confidence': 0.0,
                'preventive_methods': ["Please contact system administrator"]
            }
        
        # Decode Base64
        if image_data.startswith('data:image/'):
            image_format, base64_data = image_data.split(';base64,')
        else:
            base64_data = image_data
        
        image_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        img_array = preprocess_image_for_model(img)
        
        # TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get result
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
    """Save scan data to Firestore"""
    try:
        doc_ref = db.collection('diseasedetection').add(scan_data)
        return doc_ref[1].id
    except Exception as e:
        print(f"Database error: {e}")
        raise e

def process_image_upload(file, user_id):
    """Process file upload and return scan data"""
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
    """Process camera capture and return scan data"""
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
@login_required # PROTECTED
def scan():
    if request.method == 'POST':
        if 'image_data' in request.form and request.form['image_data']:
            try:
                # User ID check is now handled by @login_required, but keeping for safety in logic
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
@login_required # PROTECTED
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
@login_required # PROTECTED
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
@login_required # PROTECTED
def save_detection():
    """
    Save a realtime detection into a per-day 'sorts' document with subcollection 'records'.
    Each day document stores aggregated counters: accepted, rejected, total.
    Individual detections are stored under sorts/{YYYY-MM-DD}/records.
    """
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Session expired'}), 401

        if 'image_data' not in request.form or 'detection_data' not in request.form:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400

        image_data = request.form['image_data']
        detection_info = json.loads(request.form['detection_data'])

        # Save image file
        if image_data.startswith('data:image/'):
            image_format, base64_data = image_data.split(';base64,')
            image_ext = image_format.split('/')[-1]
        else:
            base64_data = image_data
            image_ext = 'jpg'

        timestamp = datetime.now()
        filename = f"realtime_detection_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.{image_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))

        # Determine date-bucket document (YYYY-MM-DD)
        date_key = timestamp.strftime('%Y-%m-%d')
        day_doc_ref = db.collection('sorts').document(date_key)

        # Prepare record
        record = {
            'user_id': session['user_id'],
            'image': os.path.join('uploads', filename).replace('\\', '/'),
            'prediction': detection_info.get('predicted_class'),
            'confidence': float(detection_info.get('confidence', 0)),
            'datetime_scanned': timestamp.isoformat()
        }

        # Save record to subcollection
        records_ref = day_doc_ref.collection('records')
        rec = records_ref.add(record)

        # Update aggregations on day document (atomic increment)
        accepted_keywords = ['half', 'healthy', 'ripe', 'over']
        pred_text = str(record['prediction'] or '').lower()
        accepted = any(k in pred_text for k in accepted_keywords)

        # Use transaction or atomic increment
        try:
            day_doc_ref.set({
                'date': date_key,
                'user_id': session['user_id']
            }, merge=True)
            # Firestore Increment
            day_doc_ref.update({
                'total': firestore.Increment(1),
                'accepted': firestore.Increment(1 if accepted else 0),
                'rejected': firestore.Increment(0 if accepted else 1)
            })
        except Exception as e:
            # If update fails because doc doesn't exist, set initial counts
            try:
                day_doc_ref.set({
                    'date': date_key,
                    'user_id': session['user_id'],
                    'total': 1,
                    'accepted': 1 if accepted else 0,
                    'rejected': 0 if accepted else 1
                }, merge=True)
            except Exception as e2:
                print(f"Error updating day summary: {e2}")

        return jsonify({
            'success': True,
            'message': 'Real-time detection saved successfully',
            'scan_id': rec[1].id,
            'date_key': date_key
        })

    except Exception as e:
        print(f"Error saving detection: {e}")
        return jsonify({
            'success': False,
            'error': f'Error saving detection: {str(e)}'
        }), 500

# -------------------- PROFILE & SETTINGS ROUTES --------------------
@app.route('/sort')
@login_required # PROTECTED
def sort():
    try:
        docs = db.collection('fruitsorter').where('user_id', '==', session['user_id']).order_by('datetime_sorted', direction=firestore.Query.DESCENDING).get()
        scans = []
        for doc in docs:
            scan_data = doc.to_dict()
            scan_data['id'] = doc.id
            scans.append(scan_data)
    except Exception as e:
        print(f"Error fetching sort data: {e}")
        scans = []
    
    return render_template('sort.html', logged_in=True, scans=scans)

@app.route('/about')
@login_required # PROTECTED
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
    # Only for Google users, use the session email (Email is not editable)
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
@login_required # PROTECTED
def profile():
    user = None
    
    if 'user_id' in session:
        try:
            user_doc = db.collection('user').document(session['user_id']).get()
            if user_doc.exists:
                user = user_doc.to_dict()
                user['id'] = user_doc.id
                
                # Ensure auth_method exists
                if 'auth_method' not in user:
                    user['auth_method'] = 'manual'
        except Exception as e:
            print(f"Error fetching user by ID: {e}")
    
    # Fallback logic from original code (preserved just in case)
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
    # Handles JSON updates to allow for partial page reloads (error messages)
    try:
        user_id = session.get('user_id')
        user_doc_ref = db.collection('user').document(user_id)
        user_doc = user_doc_ref.get()
        
        if not user_doc.exists:
            return jsonify({'success': False, 'message': 'User not found.'})
            
        user_data = user_doc.to_dict()
        auth_method = user_data.get('auth_method', 'manual')
        
        # Get Data from JSON
        data = request.get_json()
        new_name = data.get('name', '').strip()
        new_username = data.get('username', '').strip()
        
        if not new_name or not new_username:
            return jsonify({'success': False, 'message': 'Name and Username are required.'})
            
        # 1. Check Username Uniqueness (if changed)
        if new_username != user_data.get('username'):
            check = db.collection('user').where('username', '==', new_username).limit(1).get()
            if len(check) > 0:
                 return jsonify({'success': False, 'message': 'Username already taken.'})

        update_payload = {
            'name': new_name,
            'username': new_username
        }
        
        # 2. Logic based on Auth Method
        if auth_method == 'manual':
            # Manual: Check Current Password
            current_password = data.get('current_password', '')
            if not current_password:
                return jsonify({'success': False, 'message': 'Current password is required.'})
            
            if not user_data.get('password'):
                 return jsonify({'success': False, 'message': 'Account error. Contact support.'})

            if not check_password_hash(user_data.get('password', ''), current_password):
                return jsonify({'success': False, 'message': 'Incorrect current password.'})
            
            # Change Password (Optional)
            if data.get('change_password'):
                new_pass = data.get('new_password', '')
                if len(new_pass) < 6:
                    return jsonify({'success': False, 'message': 'New password must be at least 6 characters.'})
                update_payload['password'] = generate_password_hash(new_pass)
                
        elif auth_method == 'google':
            # Google: Check Verification Code
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
                
            # Clear code on success
            session.pop('profile_verification_code', None)

        # 3. Save to DB
        user_doc_ref.update(update_payload)
        
        # Update Session
        session['username'] = new_username
        
        return jsonify({'success': True, 'message': 'Save. You successfully update your profile.'})

    except Exception as e:
        print(f"Update error: {e}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

# -------------------- API ENDPOINTS --------------------

@app.route('/api/disease-data')
@login_required # PROTECTED
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
@login_required # PROTECTED
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

# -------------------- ROBUST CHATBOT ROUTE (V2 FIXED) --------------------
@app.route('/api/chat', methods=['POST'])
@login_required # PROTECTED
def chat():
    # 1. Check if Model is loaded
    if not model:
        print("❌ ERROR: Gemini Model is not configured. Check your API Key.")
        return jsonify({'reply': "System Error: API Key is missing or invalid."})

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        print(f"📩 Received message: {user_message}") 

        # 2. Get History (Memory)
        chat_history = session.get('chat_history', [])
        
        # 3. Start Chat Session
        chat_session = model.start_chat(history=chat_history)
        
        # 4. Send Message
        response = chat_session.send_message(user_message)
        
        # 5. Extract Text SAFELY
        try:
            bot_reply = response.text
        except ValueError:
            print("⚠️ WARNING: Gemini blocked the response due to safety filters.")
            bot_reply = "I apologize, but I cannot answer that query due to safety guidelines."

        print(f"🤖 Bot Reply: {bot_reply}") 

        # 6. Update History
        chat_history.append({'role': 'user', 'parts': [user_message]})
        chat_history.append({'role': 'model', 'parts': [bot_reply]})
        session['chat_history'] = chat_history[-20:] # Keep last 20 turns
        session.modified = True

        return jsonify({'reply': bot_reply})

    except Exception as e:
        print(f"❌ CRITICAL ERROR in /api/chat: {e}")
        import traceback
        traceback.print_exc() 
        return jsonify({'reply': f"Error: {str(e)}"}), 500
    
# Raspi connection config (override via .env if needed)
RASPI_HOST = os.getenv('RASPI_HOST', '10.183.102.246')  # your Pi IP
RASPI_VIDEO_PATH = os.getenv('RASPI_VIDEO_PATH', f'http://{RASPI_HOST}:5000/video_feed')
RASPI_SERVO_URL = os.getenv('RASPI_SERVO_URL', f'http://{RASPI_HOST}:5000/move_servo')

# Simple ping endpoint with diagnostic info
@app.route('/api/raspi_ping')
@login_required
def raspi_ping():
    try:
        # try a short streaming GET to confirm MJPEG endpoint and capture headers/status
        r = requests.get(RASPI_VIDEO_PATH, stream=True, timeout=5)
        status = r.status_code
        ctype = r.headers.get('Content-Type', '')
        # read a small chunk to ensure the stream responds
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

# Proxy MJPEG stream from RasPi so client can request same-origin
@app.route('/raspi_stream')
@login_required
def raspi_stream():
    def stream_generator():
        try:
            with requests.get(RASPI_VIDEO_PATH, stream=True, timeout=(3, 10)) as r:
                # stream raw chunks through
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
        except Exception as e:
            app.logger.warning(f"Error proxying raspi stream: {e}")
            # stop generator
            return
    # Use the actual content-type if available; fallback to MJPEG
    try:
        head = requests.head(RASPI_VIDEO_PATH, timeout=3)
        content_type = head.headers.get('Content-Type', 'multipart/x-mixed-replace;boundary=frame')
    except Exception:
        content_type = 'multipart/x-mixed-replace; boundary=frame'
    return Response(stream_generator(), mimetype=content_type)

# Proxy servo move commands to RasPi
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


# ==========================================
#  ANALYTICS ROUTE (VERSION 1 LOGIC RESTORED)
#  This restores the ability to use Custom Date Ranges
# ==========================================

@app.route('/api/sort-analytics')
@login_required
def sort_analytics():
    """
    Aggregates data from collection 'sorts'.
    Supports Custom Date Ranges (startDate, endDate) OR standard periods (day/week/month).
    Populates 'history' with all records in the range for reporting.
    """
    try:
        # Get parameters
        period = request.args.get('period', 'day').lower()
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')
        
        now = datetime.now()
        labels = []
        buckets = {}
        history = []

        # --- DATE LOGIC ---
        # If custom dates are provided, use them to generate chart labels
        if start_date_str and end_date_str:
            try:
                # Ensure we handle dates correctly
                current_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
                
                # Generate labels for every day in the range (for the chart x-axis)
                while current_date <= end_date_obj:
                    label = current_date.strftime('%Y-%m-%d')
                    labels.append(label)
                    buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}
                    current_date += timedelta(days=1)
            except ValueError:
                return jsonify({'error': 'Invalid date format'}), 400
        
        # Otherwise, fall back to standard period logic
        else:
            if period == 'week':
                for i in range(11, -1, -1):
                    d = now - timedelta(weeks=i)
                    y, w, _ = d.isocalendar()
                    label = f"{y}-W{w:02d}"
                    labels.append(label)
                    buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}
            elif period == 'month':
                for i in range(11, -1, -1):
                    month_date = (now.replace(day=1) - timedelta(days=30*i))
                    label = month_date.strftime('%Y-%m')
                    if label not in buckets:
                        labels.insert(0, label)
                        buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}
            else: # Default: Day (Last 30 days)
                for i in range(29, -1, -1):
                    d = now - timedelta(days=i)
                    label = d.strftime('%Y-%m-%d')
                    labels.append(label)
                    buckets[label] = {'accepted': 0, 'rejected': 0, 'total': 0}

        # --- FIREBASE QUERY ---
        sorts_ref = db.collection('sorts').where('user_id', '==', session['user_id'])
        
        # Important: Apply Date Filtering to the Query
        if start_date_str and end_date_str:
            # Firestore string comparison works for YYYY-MM-DD
            sorts_ref = sorts_ref.where('date', '>=', start_date_str).where('date', '<=', end_date_str)

        sorts_docs = sorts_ref.get()

        for day_doc in sorts_docs:
            day = day_doc.to_dict()
            date_key = day.get('date')
            if not date_key: continue

            # Determine label mapping for the chart
            label = date_key # Default for 'day' or custom range
            
            if not start_date_str: # Only remap if using standard periods
                try:
                    dt_obj = datetime.strptime(date_key, '%Y-%m-%d')
                    if period == 'week':
                        y, w, _ = dt_obj.isocalendar()
                        label = f"{y}-W{w:02d}"
                    elif period == 'month':
                        label = dt_obj.strftime('%Y-%m')
                except:
                    pass

            # Update Chart Buckets
            if label in buckets:
                buckets[label]['accepted'] += int(day.get('accepted') or 0)
                buckets[label]['rejected'] += int(day.get('rejected') or 0)
                buckets[label]['total'] += int(day.get('total') or 0)

            # --- FETCH HISTORY RECORDS ---
            try:
                # Fetch subcollection 'records' for this day
                recs = day_doc.reference.collection('records').order_by('datetime_scanned', direction=firestore.Query.DESCENDING).get()
                for r in recs:
                    rdata = r.to_dict()
                    history.append({
                        'datetime': rdata.get('datetime_scanned'),
                        'prediction': rdata.get('prediction'),
                        'confidence': float(rdata.get('confidence') or 0),
                        'accepted': any(k in str(rdata.get('prediction','')).lower() for k in ['half','healthy','ripe','over']),
                        'image': rdata.get('image')
                    })
            except Exception as e:
                print(f"Error fetching subcollection for {date_key}: {e}")

        # Final Arrays for Chart
        accepted_data = [buckets.get(l, {}).get('accepted', 0) for l in labels]
        rejected_data = [buckets.get(l, {}).get('rejected', 0) for l in labels]
        
        # Sort history by date descending
        history.sort(key=lambda x: x['datetime'], reverse=True)

        return jsonify({
            'labels': labels,
            'accepted': accepted_data,
            'rejected': rejected_data,
            'history': history 
        })

    except Exception as e:
        print(f"Error in sort_analytics: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)