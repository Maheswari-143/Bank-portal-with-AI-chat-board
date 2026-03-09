import os
import json
import csv
import re
import string
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import urlparse
import socket
import fnmatch
import pathlib
from datetime import datetime

# Initialize Flask app
templates_path = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=templates_path)
app.secret_key = 'bank_secret_key'

# Allow configuring admin panel URL via environment variable (default port 5001)
ADMIN_PANEL_URL = os.environ.get('ADMIN_PANEL_URL', 'http://localhost:8501/')

# Configure Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bank.db'
db = SQLAlchemy(app)

# Training data file path
TRAINING_DATA_FILE = os.path.join(os.path.dirname(__file__), 'training_data.json')

# Load training data
def load_training_data():
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

training_data = load_training_data()

# Load CSV dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'bankbot', 'milestone 2', 'bank_chatbot_dataset.csv')
ADMIN_DATASET_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'admin_pannel', 'bank_chatbot_dataset.csv'))
USER_DATA_FILE = os.path.join(os.path.dirname(__file__), 'user_data.json')

def load_dataset():
    def _normalize_local(s):
        if not s:
            return ''
        return ' '.join(s.lower().strip().split())

    def _load_csv(path):
        rows = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned = {}
                    for k, v in row.items():
                        cleaned[k] = (v or '').strip()
                    rows.append(cleaned)
        return rows

    merged = []
    seen = set()
    for path in [DATASET_PATH, ADMIN_DATASET_PATH]:
        for row in _load_csv(path):
            key = (
                _normalize_local(row.get('text', '')),
                row.get('intent', ''),
                row.get('entities', '')
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

dataset = load_dataset()
user_data = load_user_data()

# Normalization helper
def normalize_text(s):
    if not s:
        return ''
    s = s.lower().strip()
    s = s.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    allowed = set(string.ascii_lowercase + string.digits + ' ')
    s = ''.join(ch for ch in s if ch in allowed)
    s = ' '.join(s.split())
    return s

def find_intent_response(user_message):
    if not user_message:
        return None
    message_norm = normalize_text(user_message)

    # Hard rule for send money keywords
    if message_norm in {'send money', 'transfer money', 'pay money', 'pay'}:
        return {'intent': 'transaction', 'response': 'Please provide the recipient account number.', 'entities': ''}

    # Check for balance-related queries WITHOUT account number
    balance_keywords = ['check my balance', 'check balance', 'show balance', 'my balance', 'account balance', 'what is my balance']
    if any(kw in message_norm for kw in balance_keywords):
        # Check if account number is also provided in the same message
        acc_match = re.search(r'\b(\d{6,})\b', user_message)
        if acc_match:
            return {
                'intent': 'check_balance',
                'response': '',
                'entities': f'ACCOUNT_NUMBER:{acc_match.group(1)}'
            }
        else:
            # Ask for account number
            return {
                'intent': 'check_balance',
                'response': "Sure! What's your account number so I can verify?",
                'entities': ''
            }

    # If user provides ONLY an account number (>=6 digits) - show balance
    acc_match = re.search(r'\b(\d{6,})\b', user_message)
    if acc_match:
        acct = acc_match.group(1)
        # Check if message is ONLY the account number or contains "it's"
        tokens = message_norm.split()
        if message_norm.replace(' ', '') == acct or (len(tokens) <= 3 and acct in message_norm):
            return {
                'intent': 'check_balance',
                'response': '',
                'entities': f'ACCOUNT_NUMBER:{acct}'
            }
        # Otherwise, assume transaction intent
        else:
            return {
                'intent': 'transaction',
                'response': 'How much would you like to send?',
                'entities': f'ACCOUNT_NUMBER:{acct}'
            }

    for row in dataset:
        row_text = (row.get('text') or '').strip().lower()
        row_norm = normalize_text(row_text)
        if row_norm and row_norm == message_norm:
            return {
                'intent': row.get('intent', ''),
                'response': row.get('response', ''),
                'entities': row.get('entities', '')
            }

    message_digits = re.findall(r'\d+', user_message)
    digits_concat = ''.join(message_digits)
    for row in dataset:
        entities = (row.get('entities') or '').strip()
        if 'ACCOUNT_NUMBER' in entities or 'MONEY' in entities:
            parts = [p for p in entities.split('|') if ':' in p]
            for p in parts:
                key, val = p.split(':', 1)
                val = val.strip()
                if val and re.search(r'\d+', val):
                    if val in user_message or val in digits_concat or any(val == d for d in message_digits):
                        return {
                            'intent': row.get('intent', ''),
                            'response': row.get('response', ''),
                            'entities': row.get('entities', '')
                        }

    msg_tokens = set(message_norm.split())
    best_row = None
    best_score = 0
    for row in dataset:
        row_text = (row.get('text') or '').strip().lower()
        row_norm = normalize_text(row_text)
        if not row_norm:
            continue
        row_tokens = set(row_norm.split())
        overlap = len(msg_tokens & row_tokens)
        if overlap > best_score:
            best_score = overlap
            best_row = row
    # require meaningful overlap; otherwise treat as out_of_scope
    if best_row and best_score >= 2:
        return {
            'intent': best_row.get('intent', ''),
            'response': best_row.get('response', ''),
            'entities': best_row.get('entities', '')
        }

    return None

def extract_entities(text, entities_str):
    entities = {}
    if not entities_str:
        return entities

    for ent in entities_str.split('|'):
        if ':' in ent:
            key, val = ent.split(':', 1)
            key = key.strip().upper()
            val = val.strip()
            if key == 'ACCOUNT_NUMBER' and val:
                entities['account_number'] = val
            elif key == 'MONEY' and val:
                entities['amount'] = val
            elif key == 'PERSON' and val:
                entities['person'] = val

    if 'account_number' not in entities:
        m = re.search(r'\b(\d{6,})\b', text)
        if m:
            entities['account_number'] = m.group(1)
    if 'amount' not in entities:
        m = re.search(r'\b(\d+)\b', text)
        if m:
            entities['amount'] = m.group(1)
    if 'person' not in entities:
        m = re.search(r'\b([A-Za-z]{2,})\b', text)
        if m:
            entities['person'] = m.group(1)

    return entities

def get_intent_color(intent):
    intent_colors = {
        'greet': '#4CAF50',
        'goodbye': '#FF9800',
        'check_balance': '#2196F3',
        'transaction_inquiry': '#9C27B0',
        'loan_inquiry': '#F44336',
        'card_inquiry': '#00BCD4',
        'block_card': '#E91E63',
        'branch_locator': '#795548',
        'transfer_money': '#FF5722',
        'thanks': '#8BC34A',
        'out_of_scope': '#757575',
        # Added explicit color for new 'transaction' intent
        'transaction': '#673AB7'
    }
    return intent_colors.get(intent, '#757575')

# -----------------------------
# Database Model
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    account_number = db.Column(db.String(20), nullable=True, unique=True)
    account_type = db.Column(db.String(50), nullable=True)
    balance = db.Column(db.Float, default=0.0)

# Add Admin model so Flask can handle admin register/login
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

# New Transaction model to track money transfers
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    to_account = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='transactions')

with app.app_context():
    db.create_all()

# Helper functions to try to find an admin script in the project (used only for helpful suggestions)
def _find_local_admin_file(base_dir):
    candidates = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            lname = name.lower()
            if 'admin' in lname and lname.endswith('.py'):
                candidates.append(os.path.join(root, name))
    for p in candidates:
        if os.path.basename(p).lower() == 'admin_app.py':
            return p
    return candidates[0] if candidates else None

def _infer_admin_port_and_path(admin_file_path):
    port = None
    route_path = '/admin'
    try:
        with open(admin_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            src = f.read()
        m = re.search(r'\b(app|admin_app)\.run\([^)]*port\s*=\s*(\d+)', src)
        if m:
            port = int(m.group(2))
        else:
            m2 = re.search(r'\bPORT\s*=\s*(\d+)', src)
            if m2:
                port = int(m2.group(1))
        if re.search(r"@\w+\.route\(['\"]\/admin\/login['\"]", src):
            route_path = '/admin/login'
        elif re.search(r"@\w+\.route\(['\"]\/admin['\"]", src):
            route_path = '/admin'
    except Exception:
        pass
    if not port:
        port = 8501  # streamlit default if your admin is a streamlit app
    return port, route_path

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')

# ---------- Role Selection ----------
@app.route('/select_role')
def select_role():
    return render_template('select_role.html')

# ---------- User Register ----------
@app.route('/user/register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "User already exists! Please login."

        try:
            # Set fixed account details on register
            new_user = User(
                username=username,
                email=email,
                password=password,
                account_number='949254126359',
                account_type='savings',
                balance=5000.0
            )
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id
            session['user_type'] = 'user'
            session['username'] = new_user.username
            # Directly go to dashboard; no create_account page anymore
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            return f"An error occurred during registration: {str(e)}"
    return render_template('user_register.html')

@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            # Ensure fixed account details are set for any existing user
            if not user.account_number:
                user.account_number = '949254126359'
            if not user.account_type:
                user.account_type = 'savings'
            if user.balance is None:
                user.balance = 5000.0
            db.session.commit()

            session['user_id'] = user.id
            session['username'] = user.username
            session['user_type'] = 'user'
            return redirect(url_for('dashboard'))
        else:
            return render_template('user_login.html', error="Invalid email or password!")
    return render_template('user_login.html')


# ---------- Admin Register (Flask) ----------
@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return "All fields required", 400

        existing = Admin.query.filter((Admin.email==email) | (Admin.username==username)).first()
        if existing:
            return "Admin already exists. Please login.", 400

        try:
            new_admin = Admin(username=username, email=email, password=password)
            db.session.add(new_admin)
            db.session.commit()
            session['admin_id'] = new_admin.id
            # after successful register, redirect to admin launch (will try to open the Streamlit admin)
            return redirect(url_for('admin_launch'))
        except Exception as e:
            db.session.rollback()
            return f"Error creating admin: {e}", 500

    return render_template('admin_register.html')

# ---------- Admin Login (Flask) ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        admin = Admin.query.filter_by(email=email, password=password).first()
        if admin:
            session['admin_id'] = admin.id
            session['admin_username'] = admin.username
            return redirect(url_for('admin_launch'))
        else:
            return "Invalid admin credentials", 401
    return render_template('admin_login.html')

# ---------- Admin Launch / Redirect (tries ADMIN_PANEL_URL or suggests start command) ----------
@app.route('/admin/launch')
def admin_launch():
    # try configured ADMIN_PANEL_URL first
    parsed = urlparse(ADMIN_PANEL_URL)
    cfg_host = parsed.hostname or 'localhost'
    cfg_port = parsed.port or (443 if parsed.scheme == 'https' else 80)

    def _can_connect(host, port, timeout=2):
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return True
        except Exception:
            return False

    if _can_connect(cfg_host, cfg_port):
        return redirect(ADMIN_PANEL_URL)

    # try to discover a local admin script and infer URL
    base_dir = os.path.dirname(__file__)
    admin_file = _find_local_admin_file(base_dir)
    suggested_url = None
    suggested_cmd = None
    if admin_file:
        port_suggest, route_path = _infer_admin_port_and_path(admin_file)
        suggested_url = f'http://localhost:{port_suggest}{route_path}'
        if _can_connect('localhost', port_suggest):
            return redirect(suggested_url)
        suggested_cmd = f'cd "{os.path.dirname(admin_file)}"\npython "{os.path.basename(admin_file)}"'
        # If this is a Streamlit app, recommend streamlit run
        if os.path.basename(admin_file).lower().endswith('.py'):
            suggested_cmd = f'cd "{os.path.dirname(admin_file)}"\nstreamlit run "{os.path.basename(admin_file)}"'

    # helpful HTML if nothing reachable
    msg = f"""
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>Admin Panel Unavailable</title></head>
    <body style="font-family:Arial,Helvetica,sans-serif;color:#222;padding:24px;">
      <h2>Admin panel is not reachable</h2>
      <p>Could not connect to the admin panel at <strong>{ADMIN_PANEL_URL}</strong>.</p>
      <p>Detected admin file: <strong>{admin_file or 'None found'}</strong></p>
      <p>Suggested URL to try: <strong>{suggested_url or 'N/A'}</strong></p>
      <p>If you have an admin script, start it in a separate terminal. Example:</p>
      <pre style="background:#f5f5f7;padding:12px;border-radius:6px;">{suggested_cmd or 'No local admin script found.'}</pre>
      <p>Or set the correct URL before starting this app (Windows cmd / PowerShell):</p>
      <pre style="background:#f5f5f7;padding:12px;border-radius:6px;">set ADMIN_PANEL_URL={suggested_url or ADMIN_PANEL_URL}

# PowerShell
$env:ADMIN_PANEL_URL = '{suggested_url or ADMIN_PANEL_URL}'</pre>
      <p><a href="{url_for('select_role')}">← Back to role selection</a></p>
    </body>
    </html>
    """
    return msg, 502

# ---------- Create Account ----------
@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    return redirect(url_for('transactions'))

# ---------- Dashboard ----------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))
    user = User.query.get(session['user_id'])
    # No redirect to create_account; use fixed values if missing
    if not user.account_number:
        user.account_number = '949254126359'
    if not user.account_type:
        user.account_type = 'savings'
    if user.balance is None:
        user.balance = 5000.0
    db.session.commit()
    return render_template('dashboard.html', username=user.username, account_number=user.account_number, balance=user.balance)

# ---------- User Details ----------
@app.route('/user_details')
def user_details():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))
    user = User.query.get(session['user_id'])
    return render_template('user_details.html', user=user)

# ---------- Check Balance ----------
@app.route('/check_balance', methods=['GET', 'POST'])
def check_balance():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))

    user = User.query.get(session['user_id'])
    balance = None  # show nothing until user submits

    if request.method == 'POST':
        acc_number = (request.form.get('account_number') or '').strip()
        if acc_number == user.account_number:
            balance = user.balance
        else:
            balance = "Account not found!"

    return render_template('check_balance.html', balance=balance)

# ---------- Transactions (view and send money) ----------
@app.route('/transaction')
def transaction_landing():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"><title>Start Transaction</title>
      <style>
        body {{ margin:0; padding:0; font-family:'Segoe UI',Arial,sans-serif; background:#f7f9fb; color:#222; }}
        .wrap {{ max-width:540px; margin:60px auto; padding:24px; background:#fff; border-radius:14px; box-shadow:0 10px 26px rgba(0,0,0,.08); text-align:center; }}
        h2 {{ margin:0 0 10px; color:#2196F3; }}
        p {{ margin:0 0 16px; }}
        button {{ background:linear-gradient(135deg,#4CAF50,#2196F3); color:#fff; border:none; padding:14px 18px; border-radius:10px; font-weight:700; cursor:pointer; box-shadow:0 8px 18px rgba(0,0,0,.16); }}
        button:hover {{ transform:translateY(-1px); }}
        a {{ color:#2196F3; text-decoration:none; font-weight:600; }}
        a:hover {{ text-decoration:underline; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h2>Ready to send money?</h2>
        <p>Your current balance: <strong>5000</strong></p>
        <button onclick="window.location.href='{url_for('transactions')}'">Go to Transactions</button>
        <p style="margin-top:14px;"><a href="{url_for('dashboard')}">← Back to dashboard</a></p>
      </div>
    </body>
    """

@app.route('/transactions', methods=['GET', 'POST'])
def transactions():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))

    user = User.query.get(session['user_id'])
    msg = ''
    if request.method == 'POST':
        action = request.form.get('action', 'send')
        if action == 'clear':
            try:
                Transaction.query.filter_by(user_id=user.id).delete()
                user.balance = 5000.0  # reset to fresh balance
                db.session.commit()
                msg = 'All transactions cleared. Balance reset to 5000.'
            except Exception as e:
                db.session.rollback()
                msg = f'Error clearing transactions: {e}'
        else:
            to_account = (request.form.get('to_account') or '').strip()
            amount_raw = (request.form.get('amount') or '').strip()
            try:
                amount = float(amount_raw)
            except Exception:
                amount = -1

            if not re.fullmatch(r'\d{6,}', to_account):
                msg = 'Invalid recipient account number.'
            elif amount <= 0:
                msg = 'Invalid amount.'
            elif user.balance < amount:
                msg = 'Insufficient balance.'
            else:
                try:
                    tx = Transaction(user_id=user.id, to_account=to_account, amount=amount)
                    user.balance = float(user.balance) - amount
                    db.session.add(tx)
                    db.session.commit()
                    msg = f'Sent {amount} to {to_account}. Available balance: {user.balance}.'
                except Exception as e:
                    db.session.rollback()
                    msg = f'Error creating transaction: {e}'

    txs = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.timestamp.desc()).all()
    items = ''.join(
        f'<li>{t.timestamp.strftime("%Y-%m-%d %H:%M:%S")} → {t.to_account} : {t.amount}</li>'
        for t in txs
    )
    display_msg = msg or f'Available balance: {user.balance}'
    status_class = 'ok' if msg.startswith('Sent') else ('err' if msg else 'info')
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"><title>Transactions</title>
      <style>
        :root {{ --primary:#2196F3; --accent:#2196F3; --bg:#f4f7f5; --card:#ffffff; --text:#222; }}
        * {{ box-sizing:border-box; }}
        body {{ margin:0; padding:0; font-family:'Segoe UI',Arial,sans-serif; background:var(--bg); color:var(--text); }}
        .header {{ background:linear-gradient(120deg,var(--primary),var(--accent)); color:#fff; padding:18px 24px; box-shadow:0 4px 12px rgba(0,0,0,.12); }}
        .container {{ max-width:820px; margin:32px auto; padding:0 18px; }}
        .card {{ background:var(--card); border-radius:12px; padding:20px; box-shadow:0 8px 24px rgba(0,0,0,.08); margin-bottom:18px; }}
        h1 {{ margin:0 0 6px; font-size:24px; }}
        h3 {{ margin:0 0 10px; }}
        label {{ font-weight:600; color:#37474f; }}
        .input {{ width:100%; padding:12px 14px; border:1px solid #d9e2ec; border-radius:8px; margin:6px 0 14px; background:#fafbfd; }}
        .btn {{ background:#2196F3; color:#fff; border:none; padding:12px 18px; border-radius:10px; cursor:pointer; font-weight:700; box-shadow:0 4px 10px rgba(33,150,243,.35); transition:transform .08s ease, box-shadow .12s ease; }}
        .btn:hover {{ transform:translateY(-1px); box-shadow:0 6px 14px rgba(33,150,243,.4); }}
        .btn.secondary {{ background:#1976D2; box-shadow:0 4px 10px rgba(25,118,210,.32); }}
        .flex {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
        .pill {{ display:inline-block; padding:8px 16px; border-radius:20px; background:#2196F3; color:#ffffff; font-weight:700; font-size:14px; }}
        .msg {{ margin:0 0 12px; font-weight:600; }}
        .msg.ok {{ color:#1b5e20; }}
        .msg.err {{ color:#c62828; }}
        .msg.info {{ color:#1565c0; }}
        ul {{ padding-left:18px; margin:10px 0; }}
        li {{ margin:8px 0; }}
        .footer-links a {{ color:var(--accent); font-weight:600; text-decoration:none; }}
        .footer-links a:hover {{ text-decoration:underline; }}
      </style>
    </head>
    <body>
      <div class="header">
        <h1>Transactions</h1>
        <div class="pill">Account {user.account_number}</div>
      </div>
      <div class="container">
        <div class="card">
          <h3>Send Money</h3>
          <p class="msg {status_class}">{display_msg}</p>
          <form method="post">
            <label>Recipient Account Number</label>
            <input class="input" type="text" name="to_account" placeholder="e.g. 123456789012" required />
            <label>Amount</label>
            <input class="input" type="number" step="0.01" min="0.01" name="amount" placeholder="e.g. 100" required />
            <div class="flex">
              <button type="submit" class="btn">Send Money</button>
              <button type="submit" name="action" value="clear" class="btn secondary" onclick="return confirm('Clear all transactions and reset balance?');">Clear All</button>
              <div class="pill">Available: {user.balance}</div>
            </div>
          </form>
        </div>

        <div class="card">
          <h3>History</h3>
          <ul>{items or '<li>No transactions yet.</li>'}</ul>
        </div>

        <div class="flex footer-links">
          <a href="{url_for('dashboard')}">← Back to dashboard</a>
        </div>
      </div>
    </body>
    </html>
    """
    return html

# ---------- Bank Bot ----------
@app.route('/bankbot')
def bankbot():
    if 'user_id' not in session:
        return redirect(url_for('user_login'))
    return render_template('bankbot.html', username=session['username'])

def append_to_dataset_row(text, intent, response, entities_str=''):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    file_exists = os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0

    for row in dataset:
        if row.get('text') == text and row.get('intent') == intent and row.get('entities') == entities_str:
            return False

    with open(DATASET_PATH, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['text','intent','response','entities'])
        writer.writerow([text, intent, response, entities_str])

    dataset.append({'text': text, 'intent': intent, 'response': response, 'entities': entities_str})
    return True

# Seed chatbot training for "send money" and sample account numbers
def seed_transaction_training():
    examples = [
        ('send money', 'transaction', 'Please provide the recipient account number.', ''),
        ('transfer money', 'transaction', 'Please provide the recipient account number.', ''),
        ('account number 123456789012', 'transaction', 'How much would you like to send?', 'ACCOUNT_NUMBER:123456789012'),
        ('account number 222222222222', 'transaction', 'How much would you like to send?', 'ACCOUNT_NUMBER:222222222222'),
        ('account number 333333333333', 'transaction', 'How much would you like to send?', 'ACCOUNT_NUMBER:333333333333'),
        ('account number 949254126359', 'transaction', 'How much would you like to send?', 'ACCOUNT_NUMBER:949254126359'),
    ]
    for text, intent, response, ents in examples:
        try:
            append_to_dataset_row(text, intent, response, ents)
        except Exception:
            pass

# Replace before_first_request (removed in Flask 3) with a guarded before_request
_seed_done = False
@app.before_request
def _ensure_seed_once():
    global _seed_done
    if not _seed_done:
        try:
            seed_transaction_training()
        except Exception:
            pass
        _seed_done = True

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return {'error': 'Unauthorized'}, 401

    # Reload dataset to pick up any new admin additions
    global dataset
    dataset = load_dataset()

    user_message = request.json.get('message', '').strip()
    user = User.query.get(session['user_id'])
    user_id_str = str(session['user_id'])

    if user_id_str not in user_data:
        user_data[user_id_str] = {
            'account_number': user.account_number,
            'balance': user.balance,
            'conversations': []
        }

    result = find_intent_response(user_message)
    intent = 'out_of_scope'
    intent_color = get_intent_color(intent)
    entities = {}
    bot_reply = ''

    if result:
        intent = result.get('intent', 'out_of_scope')
        intent_color = get_intent_color(intent)
        entities = extract_entities(user_message, result.get('entities', ''))
        bot_reply = (result.get('response') or '').strip()
        
        # Special handling for check_balance intent
        if intent == 'check_balance':
            if entities.get('account_number'):
                acct_num = entities['account_number']
                # Check if the account matches logged-in user
                if acct_num == user.account_number:
                    bot_reply = f"💰 Your available balance is ₹{user.balance}"
                else:
                    bot_reply = "❌ Sorry, I can only show balance for your registered account."
            # If no account number, bot_reply already has "Sure! What's your account number..."
        
        if not bot_reply:
            if entities.get('amount'):
                bot_reply = f"💰 Your balance is {entities['amount']}."
            else:
                acct = entities.get('account_number', '')
                if acct:
                    for row in dataset:
                        ents = (row.get('entities') or '')
                        if f"ACCOUNT_NUMBER:{acct}" in ents and 'MONEY:' in ents:
                            m = re.search(r'MONEY:(\d+)', ents)
                            if m:
                                bot_reply = f"💰 Your balance is {m.group(1)}."
                                entities['amount'] = m.group(1)
                                break
        if not bot_reply:
            reply_digits = re.findall(r'\d+', user_message)
            if reply_digits:
                bot_reply = f"💰 Your balance is {reply_digits[0]}."
        if bot_reply is None:
            bot_reply = ''

        if entities:
            user_data[user_id_str].update(entities)
            if 'amount' in entities:
                user_data[user_id_str]['last_amount'] = entities['amount']
            if 'person' in entities:
                user_data[user_id_str]['last_recipient'] = entities['person']
            if 'account_number' in entities:
                user_data[user_id_str]['account_number'] = entities['account_number']

        user_data[user_id_str]['conversations'].append({
            'user': user_message,
            'bot': bot_reply,
            'intent': intent
        })
        save_user_data(user_data)

        add_entities = []
        if 'amount' in entities:
            add_entities.append(f"MONEY:{entities['amount']}")
        if 'account_number' in entities:
            add_entities.append(f"ACCOUNT_NUMBER:{entities['account_number']}")
        entities_str = '|'.join(add_entities)

        reply_digits = re.findall(r'\d+', str(bot_reply))
        if not entities_str and reply_digits:
            entities_str = f"MONEY:{reply_digits[0]}"

        if entities_str:
            try:
                append_to_dataset_row(user_message, intent, bot_reply, entities_str)
            except Exception:
                pass
    else:
        bot_reply = "I can only assist with banking questions. Try asking about balance, transfers, loans, or cards."
        intent = "out_of_scope"
        intent_color = get_intent_color(intent)
        user_data[user_id_str]['conversations'].append({
            'user': user_message,
            'bot': bot_reply,
            'intent': intent
        })
        save_user_data(user_data)

    return {
        'reply': bot_reply,
        'intent': intent,
        'intent_color': intent_color
    }

# ---------- Logout ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Add a health check endpoint
@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': str(datetime.now())}, 200




if __name__ == '__main__':
    app.run(debug=True)



