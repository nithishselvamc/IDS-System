from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)

# ── Load model, feature list, path encoder ─────────────────────────────────────
model    = joblib.load("rf_model.pkl")
labels   = joblib.load("feature_columns.pkl")
le_path  = joblib.load("le_path.pkl")

# ── Feature helpers ────────────────────────────────────────────────────────────
def count_dot(s):           return s.count('.')
def no_of_dir(s):           return urlparse(s).path.count('/')
def no_of_embed(s):         return urlparse(s).path.count('//')
def count_http(s):          return s.count('http')
def count_per(s):           return s.count('%')
def count_ques(s):          return s.count('?')
def count_hyphen(s):        return s.count('-')
def count_equal(s):         return s.count('=')
def url_length(s):          return len(str(s))
def hostname_length(s):     return len(urlparse(s).netloc)
def digit_count(s):         return sum(1 for c in s if c.isnumeric())
def letter_count(s):        return sum(1 for c in s if c.isalpha())
def count_special(s):       return len(re.sub(r'[a-zA-Z0-9\s]', '', s))
def number_of_parameters(s):
    p = urlparse(s).query
    return 0 if p == '' else len(p.split('&'))
def is_encoded(s):          return int('%' in s.lower())
def unusual_ratio(s):
    total = len(s)
    unusual = re.sub(r'[a-zA-Z0-9\s\-._]', '', s)
    return len(unusual) / total if total > 0 else 0

# ── Fixed suspicious words (no false positives for normal form fields) ─────────
SCORE_MAP = {
    'select': 50, 'from': 50, 'where': 50, 'delete': 50,
    'drop': 50, 'create': 50, 'table': 50, 'union': 35,
    'like': 30, '--': 30,
    'alert': 30, 'javascript': 20, 'script': 25, 'iframe': 25,
    'onerror': 30, 'document.cookie': 40, 'set-cookie': 40,
    'cmd': 40, 'shell': 40, 'exec': 30, 'eval': 30, 'ssh': 40,
    '../': 40, '.exe': 30, '.php': 20,
    'malware': 45, 'exploit': 45, 'backdoor': 45,
    'inject': 30, 'phishing': 45, 'rootkit': 45,
    'cookiesteal': 40, 'document.': 20,
    'include': 30, 'fetch': 25,
}

def suspicious_words(s):
    matches = re.findall(r'(?i)' + '|'.join(re.escape(k) for k in SCORE_MAP.keys()), s)
    return sum(SCORE_MAP.get(m.lower(), 0) for m in matches)

def apply_content(content, fn):
    if not content or not isinstance(content, str):
        return 0
    return fn(content)

METHOD_MAP = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3, 'HEAD': 4, 'OPTIONS': 5}

# ── Clean URL path (strip HTTP/1.1 artifact from CSIC dataset) ────────────────
def clean_path(url):
    path = urlparse(url).path
    path = re.sub(r'\s*HTTP/\d+\.\d+.*$', '', path).strip()
    return path

def get_path_enc(url):
    path = clean_path(url)
    if path in le_path.classes_:
        return int(le_path.transform([path])[0])
    # Unknown path — use fallback index 0
    return 0

# ── Build feature vector ───────────────────────────────────────────────────────
def extract_features(form_data):
    url     = form_data.get('url', '')
    content = form_data.get('content', '')
    method  = form_data.get('method', 'GET')

    feat = {
        'count_dot_url':               count_dot(url),
        'count_dir_url':               no_of_dir(url),
        'count_embed_domain_url':      no_of_embed(url),
        'count-http':                  count_http(url),
        'count%_url':                  count_per(url),
        'count?_url':                  count_ques(url),
        'count-_url':                  count_hyphen(url),
        'count=_url':                  count_equal(url),
        'url_length':                  url_length(url),
        'hostname_length_url':         hostname_length(url),
        'sus_url':                     suspicious_words(url),
        'count-digits_url':            digit_count(url),
        'count-letters_url':           letter_count(url),
        'number_of_parameters_url':    number_of_parameters(url),
        'is_encoded_url':              is_encoded(url),
        'special_count_url':           count_special(url),
        'unusual_character_ratio_url': unusual_ratio(url),
        'Method_enc':                  METHOD_MAP.get(method.upper(), 0),
        'count_dot_content':           apply_content(content, count_dot),
        'count%_content':              apply_content(content, count_per),
        'count-_content':              apply_content(content, count_hyphen),
        'count=_content':              apply_content(content, count_equal),
        'sus_content':                 apply_content(content, suspicious_words),
        'count_digits_content':        apply_content(content, digit_count),
        'count_letters_content':       apply_content(content, letter_count),
        'content_length':              apply_content(content, url_length),
        'is_encoded_content':          apply_content(content, is_encoded),
        'special_count_content':       apply_content(content, count_special),
        'url_path_enc':                get_path_enc(url),
    }

    vector = np.array([[feat[col] for col in labels]], dtype=float)
    return vector, feat

# ── Verdict logic with 3-tier confidence threshold ────────────────────────────
#   attack_proba >= 0.80  → ATTACK    (high confidence)
#   attack_proba >= 0.55  → SUSPICIOUS (uncertain)
#   attack_proba <  0.55  → NORMAL
def get_verdict(attack_proba):
    if attack_proba >= 0.80:
        return 'attack',     'ATTACK DETECTED'
    elif attack_proba >= 0.55:
        return 'suspicious', 'SUSPICIOUS'
    else:
        return 'normal',     'NORMAL TRAFFIC'

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data            = request.get_json()
    vector, feat    = extract_features(data)
    pred            = model.predict(vector)[0]
    proba           = model.predict_proba(vector)[0].tolist()
    classes         = model.classes_.tolist()

    # Map class indices to labels
    class_labels = []
    for c in classes:
        class_labels.append('Normal' if str(c) in ['0', 'Normal', 'normal'] else 'Anomalous')

    prob_dict      = {lbl: round(p * 100, 2) for lbl, p in zip(class_labels, proba)}
    attack_proba   = prob_dict.get('Anomalous', 0) / 100.0
    tier, verdict  = get_verdict(attack_proba)

    result = {
        'prediction':    int(pred),
        'tier':          tier,           # 'normal' | 'suspicious' | 'attack'
        'label':         verdict,
        'attack_proba':  round(attack_proba * 100, 2),
        'probabilities': prob_dict,
        'features':      {k: round(float(v), 4) for k, v in feat.items()},
        'url_path':      clean_path(data.get('url', '')),
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)