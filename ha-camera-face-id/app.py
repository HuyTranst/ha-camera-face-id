import os
import sys
import json
import time
import sqlite3
import threading
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import paho.mqtt.client as mqtt
import insightface
from insightface.app import FaceAnalysis

# Configuration paths for Home Assistant Add-on
DATA_DIR = "/data"
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

OPTIONS_PATH = os.path.join(DATA_DIR, "options.json")

# Default settings
CONFIG = {
    "mqtt_host": "core-mosquitto",
    "mqtt_port": 1883,
    "mqtt_user": "",
    "mqtt_password": "",
    "similarity_threshold": 0.55,
    "min_face_size": 50,
    "scan_interval_seconds": 2
}

if os.path.exists(OPTIONS_PATH):
    try:
        with open(OPTIONS_PATH, "r") as f:
            user_opts = json.load(f)
            CONFIG.update(user_opts)
    except Exception as e:
        print(f"[Config] Error loading options.json: {e}")

# Directories for faces & events
FACES_DIR = os.path.join(DATA_DIR, "faces")
EVENTS_DIR = os.path.join(DATA_DIR, "events")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database Setup
DB_PATH = os.path.join(DATA_DIR, "face_id.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT UNIQUE NOT NULL,
            name TEXT,
            enabled INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_entity_id TEXT NOT NULL,
            person_name TEXT NOT NULL,
            similarity REAL NOT NULL,
            crop_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Initialize InsightFace Model
print("[AI] Initializing InsightFace model (buffalo_sc)...")
face_app = None
try:
    face_app = FaceAnalysis(name='buffalo_sc', root=MODELS_DIR, providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[AI] InsightFace initialized successfully!")
except Exception as e:
    print(f"[AI] Error initializing InsightFace: {e}")

# Helper: Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# Load all trained face embeddings from DB
def load_trained_embeddings():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.name, fs.embedding 
        FROM face_samples fs 
        JOIN persons p ON fs.person_id = p.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    embeddings = []
    for r in rows:
        emb_array = np.array(json.loads(r['embedding']), dtype=np.float32)
        embeddings.append((r['name'], emb_array))
    return embeddings

# MQTT Setup
mqtt_client = mqtt.Client()
mqtt_connected = False

def connect_mqtt():
    global mqtt_connected
    if not CONFIG.get("mqtt_host"):
        return
    try:
        if CONFIG.get("mqtt_user"):
            mqtt_client.username_pw_set(CONFIG["mqtt_user"], CONFIG.get("mqtt_password", ""))
        mqtt_client.connect(CONFIG["mqtt_host"], int(CONFIG.get("mqtt_port", 1883)), 60)
        mqtt_client.loop_start()
        mqtt_connected = True
        print(f"[MQTT] Connected to {CONFIG['mqtt_host']}:{CONFIG['mqtt_port']}")
    except Exception as e:
        print(f"[MQTT] Failed to connect: {e}")

connect_mqtt()

def publish_mqtt_discovery(camera_slug):
    if not mqtt_connected:
        return
    topic = f"homeassistant/sensor/ha_face_id_{camera_slug}/config"
    payload = {
        "name": f"Face ID {camera_slug}",
        "state_topic": f"ha_camera_face_id/{camera_slug}/state",
        "value_template": "{{ value_json.person }}",
        "json_attributes_topic": f"ha_camera_face_id/{camera_slug}/state",
        "icon": "mdi:face-recognition",
        "unique_id": f"ha_face_id_{camera_slug}"
    }
    mqtt_client.publish(topic, json.dumps(payload), retain=True)

# Fetch Camera Image from HA API
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

def fetch_camera_snapshot(entity_id):
    headers = {
        "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
        "Content-Type": "application/json"
    }
    url = f"http://supervisor/core/api/camera_proxy/{entity_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            image_bytes = resp.content
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        pass
    return None

# Background Camera Worker Loop
def camera_worker_loop():
    print("[Worker] Camera AI scanner worker started.")
    while True:
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT entity_id, name FROM active_cameras WHERE enabled = 1")
            cameras = cursor.fetchall()
            conn.close()

            if cameras and face_app:
                trained_embeddings = load_trained_embeddings()
                min_face = CONFIG.get("min_face_size", 50)
                threshold = CONFIG.get("similarity_threshold", 0.55)

                for cam in cameras:
                    entity_id = cam["entity_id"]
                    cam_slug = entity_id.replace(".", "_")
                    publish_mqtt_discovery(cam_slug)

                    img = fetch_camera_snapshot(entity_id)
                    if img is None:
                        continue

                    # Run InsightFace
                    faces = face_app.get(img)
                    if not faces:
                        continue

                    for idx, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        if w < min_face or h < min_face:
                            continue

                        embedding = face.embedding
                        best_match_name = "Unknown"
                        best_sim = 0.0

                        for name, trained_emb in trained_embeddings:
                            sim = cosine_similarity(embedding, trained_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_match_name = name

                        if best_sim < threshold:
                            best_match_name = "Unknown"

                        # Crop face image
                        pad = 10
                        x1 = max(0, bbox[0] - pad)
                        y1 = max(0, bbox[1] - pad)
                        x2 = min(img.shape[1], bbox[2] + pad)
                        y2 = min(img.shape[0], bbox[3] + pad)
                        crop_img = img[y1:y2, x1:x2]

                        timestamp_str = str(int(time.time() * 1000))
                        crop_filename = f"event_{timestamp_str}_{idx}.jpg"
                        crop_path = os.path.join(EVENTS_DIR, crop_filename)
                        cv2.imwrite(crop_path, crop_img)

                        # Save Event to DB
                        conn = get_db()
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO detection_events (camera_entity_id, person_name, similarity, crop_path)
                            VALUES (?, ?, ?, ?)
                        ''', (entity_id, best_match_name, round(best_sim, 2), crop_filename))
                        conn.commit()
                        conn.close()

                        # Publish MQTT State
                        if mqtt_connected:
                            mqtt_payload = {
                                "person": best_match_name,
                                "similarity": round(best_sim, 2),
                                "camera": entity_id,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "crop_file": crop_filename
                            }
                            mqtt_client.publish(f"ha_camera_face_id/{cam_slug}/state", json.dumps(mqtt_payload))

        except Exception as e:
            print(f"[Worker Error] {e}")

        time.sleep(CONFIG.get("scan_interval_seconds", 2))

# Start background thread
worker_thread = threading.Thread(target=camera_worker_loop, daemon=True)
worker_thread.start()

# Flask Web UI Server
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/events/<filename>')
def serve_event_img(filename):
    return send_from_directory(EVENTS_DIR, filename)

@app.route('/static/faces/<filename>')
def serve_face_img(filename):
    return send_from_directory(FACES_DIR, filename)

@app.route('/api/ha_cameras')
def get_ha_cameras():
    headers = {"Authorization": f"Bearer {SUPERVISOR_TOKEN}"}
    try:
        resp = requests.get("http://supervisor/core/api/states", headers=headers, timeout=5)
        if resp.status_code == 200:
            states = resp.json()
            cameras = [
                {
                    "entity_id": s["entity_id"],
                    "name": s.get("attributes", {}).get("friendly_name", s["entity_id"]),
                    "state": s.get("state")
                }
                for s in states if s["entity_id"].startswith("camera.")
            ]
            return jsonify(cameras)
    except Exception as e:
        pass
    return jsonify([])

@app.route('/api/active_cameras', methods=['GET', 'POST', 'DELETE'])
def manage_active_cameras():
    conn = get_db()
    cursor = conn.cursor()
    if request.method == 'GET':
        cursor.execute("SELECT * FROM active_cameras")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify(rows)

    elif request.method == 'POST':
        data = request.json
        entity_id = data.get('entity_id')
        name = data.get('name', entity_id)
        cursor.execute("INSERT OR REPLACE INTO active_cameras (entity_id, name, enabled) VALUES (?, ?, 1)", (entity_id, name))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    elif request.method == 'DELETE':
        entity_id = request.args.get('entity_id')
        cursor.execute("DELETE FROM active_cameras WHERE entity_id = ?", (entity_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

@app.route('/api/persons', methods=['GET', 'POST', 'DELETE'])
def manage_persons():
    conn = get_db()
    cursor = conn.cursor()
    if request.method == 'GET':
        cursor.execute('''
            SELECT p.id, p.name, COUNT(fs.id) as sample_count 
            FROM persons p 
            LEFT JOIN face_samples fs ON p.id = fs.person_id 
            GROUP BY p.id
        ''')
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify(rows)

    elif request.method == 'POST':
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({"error": "Tên không được để trống"}), 400

        cursor.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
        conn.commit()
        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        person_id = cursor.fetchone()['id']

        files = request.files.getlist('photos')
        uploaded = 0
        for file in files:
            if file and face_app:
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    faces = face_app.get(img)
                    if faces:
                        # Extract embedding from largest face
                        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                        embedding_json = json.dumps(largest_face.embedding.tolist())
                        
                        filename = f"person_{person_id}_{int(time.time()*1000)}.jpg"
                        filepath = os.path.join(FACES_DIR, filename)
                        cv2.imwrite(filepath, img)

                        cursor.execute("INSERT INTO face_samples (person_id, file_path, embedding) VALUES (?, ?, ?)",
                                       (person_id, filename, embedding_json))
                        uploaded += 1

        conn.commit()
        conn.close()
        return jsonify({"success": True, "uploaded": uploaded})

    elif request.method == 'DELETE':
        person_id = request.args.get('id')
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

@app.route('/api/events')
def get_events():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detection_events ORDER BY id DESC LIMIT 50")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/test_face_id', methods=['POST'])
def test_face_id():
    data = request.json
    entity_id = data.get('entity_id')
    if not entity_id or not face_app:
        return jsonify({"error": "Camera entity hoặc AI model chưa sẵn sàng"}), 400

    img = fetch_camera_snapshot(entity_id)
    if img is None:
        return jsonify({"error": "Không thể lấy ảnh snapshot từ camera này"}), 400

    faces = face_app.get(img)
    if not faces:
        return jsonify({"message": "Không tìm thấy khuôn mặt nào trong ảnh snapshot"}), 200

    trained_embeddings = load_trained_embeddings()
    threshold = CONFIG.get("similarity_threshold", 0.55)
    results = []

    for face in faces:
        embedding = face.embedding
        best_name = "Unknown"
        best_sim = 0.0

        for name, trained_emb in trained_embeddings:
            sim = cosine_similarity(embedding, trained_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            best_name = "Unknown"

        results.append({
            "name": best_name,
            "similarity": round(best_sim, 2),
            "score": round(float(face.det_score), 2),
            "bbox": face.bbox.astype(int).tolist()
        })

    return jsonify({"faces_found": len(faces), "results": results})

if __name__ == '__main__':
    print("[Web UI] Starting HA Camera Face ID Web Server on port 8099...")
    app.run(host='0.0.0.0', port=8099)
