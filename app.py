"""
Smart Door Lock System - Main Flask Application
Face Recognition & Emotion Detection Dashboard
"""

import os
import base64
import numpy as np
import cv2
from datetime import datetime, date
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_file
)

from models import (
    init_db, verify_admin,
    add_class, get_all_classes, delete_class, get_class_by_id,
    add_student, get_all_students, get_student_by_id,
    get_students_by_class, delete_student,
    add_access_log, get_access_logs,
    add_alert, get_active_alerts, resolve_all_alerts,
    get_db,
)
from recognition_engine import (
    save_face_image, recognize_face,
    get_emotion_emoji, get_emotion_label_id,
    get_model_info,
)
from robotics_controller import door_controller

# ─── Flask App ───────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "smartdoor-secret-key-2026"

# Initialize database on startup
init_db()
print("✅ Database initialized.")


# ─── Auth Helper ─────────────────────────────────────────────────

def login_required(f):
    """Decorator to require admin login."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

# ─── Login / Logout ──────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if verify_admin(username, password):
            session["logged_in"] = True
            session["username"] = username
            flash("Login successful! Welcome, Admin.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# ─── Dashboard ───────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    students = get_all_students()
    classes = get_all_classes()
    logs = get_access_logs(limit=50)
    alerts = get_active_alerts()

    # Count today's access
    today_str = date.today().isoformat()
    today_access = 0
    for log in logs:
        if log["timestamp"] and today_str in log["timestamp"]:
            today_access += 1

    # Mood counts from today's logs
    mood_counts = {}
    for log in logs:
        if log["timestamp"] and today_str in log["timestamp"]:
            mood = log["detected_mood"]
            if mood:
                mood_lower = mood.split(" ")[0].lower()  # e.g. "😊 happy" -> "happy"
                mood_counts[mood_lower] = mood_counts.get(mood_lower, 0) + 1

    return render_template(
        "dashboard.html",
        total_students=len(students),
        total_classes=len(classes),
        today_access=today_access,
        active_alerts=len(alerts),
        recent_logs=logs,
        mood_counts=mood_counts,
    )


# ─── Class Management ───────────────────────────────────────────

@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes():
    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        description = request.form.get("description", "").strip()
        start_time = request.form.get("start_time", "").strip()
        end_time = request.form.get("end_time", "").strip()
        if class_name:
            if add_class(class_name, description, start_time, end_time):
                flash(f"Class '{class_name}' added successfully!", "success")
            else:
                flash(f"Class '{class_name}' already exists.", "error")
        else:
            flash("Class name cannot be empty.", "error")
        return redirect(url_for("classes"))

    all_classes = get_all_classes()
    return render_template("classes.html", classes=all_classes)


@app.route("/classes/delete/<int:class_id>", methods=["POST"])
@login_required
def delete_class_route(class_id):
    delete_class(class_id)
    flash("Class deleted.", "success")
    return redirect(url_for("classes"))


# ─── Student Management ─────────────────────────────────────────

@app.route("/students")
@login_required
def students():
    all_students = get_all_students()
    return render_template("students.html", students=all_students)


@app.route("/add_student", methods=["GET", "POST"])
@login_required
def add_student_route():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        class_id = request.form.get("class_id", "")
        face_data = request.form.get("face_data", "")

        if not student_id or not name or not class_id:
            flash("Please fill all fields.", "error")
            return redirect(url_for("add_student_route"))

        if not face_data:
            flash("Please capture a face photo.", "error")
            return redirect(url_for("add_student_route"))

        # Decode base64 face image
        try:
            header, encoded = face_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            face_path = save_face_image(student_id, name, img_bytes)
            if not face_path:
                flash("Failed to save face image.", "error")
                return redirect(url_for("add_student_route"))
        except Exception as e:
            flash(f"Error processing face image: {e}", "error")
            return redirect(url_for("add_student_route"))

        if add_student(student_id, name, int(class_id), face_path):
            flash(f"Student '{name}' (ID: {student_id}) registered successfully!", "success")
            # Clear DeepFace cache so new face is recognized
            _clear_deepface_cache()
            return redirect(url_for("students"))
        else:
            flash(f"Student ID '{student_id}' already exists.", "error")
            return redirect(url_for("add_student_route"))

    all_classes = get_all_classes()
    return render_template("add_student.html", classes=all_classes)


@app.route("/students/delete/<int:student_db_id>", methods=["POST"])
@login_required
def delete_student_route(student_db_id):
    delete_student(student_db_id)
    _clear_deepface_cache()
    flash("Student deleted.", "success")
    return redirect(url_for("students"))


@app.route("/face_image/<int:student_db_id>")
def face_image(student_db_id):
    """Serve a student's face image."""
    from recognition_engine import DATASET_DIR
    student = get_student_by_id(student_db_id)
    if student and student["face_image_path"]:
        db_path = student["face_image_path"]
        if os.path.exists(db_path):
            return send_file(db_path, mimetype="image/jpeg")
            
        # Fallback: dynamically rebuild the path relative to the current DATASET_DIR.
        # This handles the case where the project folder was moved from another PC.
        base_name = os.path.basename(db_path)
        dir_name = os.path.basename(os.path.dirname(db_path))
        fallback_path = os.path.join(DATASET_DIR, dir_name, base_name)
        
        if os.path.exists(fallback_path):
            return send_file(fallback_path, mimetype="image/jpeg")
            
    return "", 404


# ─── Access Logs ─────────────────────────────────────────────────

@app.route("/logs")
@login_required
def logs():
    all_logs = get_access_logs(limit=200)
    return render_template("logs.html", logs=all_logs)


# ─── Door Camera Page ───────────────────────────────────────────

@app.route("/door")
@login_required
def door():
    all_classes = get_all_classes()
    return render_template("door_camera.html", classes=all_classes)


# ═══════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    Receive a base64 JPEG from the Door Camera page,
    run face recognition + emotion detection,
    and return access decision.
    """
    data = request.get_json()
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"error": "No image data", "access": "denied"}), 400

    try:
        # Decode base64
        try:
            header, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Image decode error: {e}"}), 400
    
        # Run recognition
        result = recognize_face(frame)
    
        if result is None:
            # No face detected
            add_access_log(None, "No face", "-", "-", "Denied", "Locked")
            door_controller.deny_access("No face detected")
            return jsonify({
                "access": "denied",
                "reason": "No face detected in the image. Please face the camera directly.",
                "student_name": None,
                "mood": None,
                "mood_emoji": None,
                "mood_label": None,
                "emotions": None,
            })
    
        mood = str(result["emotion"])
        mood_emoji = get_emotion_emoji(mood)
        mood_label = get_emotion_label_id(mood)
        emotions = {str(k): float(v) for k, v in result.get("emotions", {}).items()}
    
        if result["recognized"]:
            # Found a matching face - look up student in DB
            folder_name = result["student_folder"]  # e.g. "STU001_John_Doe"
            parts = folder_name.split("_", 1)
            matched_student_id = parts[0] if parts else ""
    
            # Find student by student_id
            conn = get_db()
            student_row = conn.execute(
                """SELECT s.*, c.class_name, c.id as cid
                   FROM students s JOIN classes c ON s.class_id = c.id
                   WHERE s.student_id = ?""",
                (matched_student_id,)
            ).fetchone()
            conn.close()
    
            if student_row:
                student_name = student_row["name"]
                student_class = student_row["class_name"]
                student_class_id = str(student_row["cid"])
    
                # ACCESS GRANTED — no class filter needed!
                door_controller.open_door(
                    reason=f"Face matched ({student_class})",
                    student_name=student_name,
                    student_id=matched_student_id,
                    mood=mood_label
                )
                add_access_log(
                    student_row["id"], student_name, student_class,
                    f"{mood_emoji} {mood}", "Granted", "Opened"
                )
                return jsonify({
                    "access": "granted",
                    "student_name": student_name,
                    "student_id": matched_student_id,
                    "class_name": student_class,
                    "mood": mood,
                    "mood_emoji": mood_emoji,
                    "mood_label": mood_label,
                    "emotions": emotions,
                })
            else:
                # Face matched a folder but student not in DB
                door_controller.deny_access("Face folder found but no DB record")
                add_access_log(None, folder_name, "-", f"{mood_emoji} {mood}", "Denied", "Locked")
                return jsonify({
                    "access": "denied",
                    "reason": "Face matched a dataset but student record not found in database.",
                    "student_name": folder_name,
                    "mood": mood,
                    "mood_emoji": mood_emoji,
                    "mood_label": mood_label,
                    "emotions": emotions,
                })
        else:
            # Face detected but not recognized
            door_controller.deny_access("Face not in database")
            add_access_log(None, "Unknown", "-", f"{mood_emoji} {mood}", "Denied", "Locked")
            return jsonify({
                "access": "denied",
                "reason": "Face not recognized. This person is not registered.",
                "student_name": "Unknown",
                "mood": mood,
                "mood_emoji": mood_emoji,
                "mood_label": mood_label,
                "emotions": emotions,
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "access": "denied",
            "reason": f"System error occurred: {str(e)}",
            "error": str(e)
        }), 500


@app.route("/api/fire_alarm", methods=["POST"])
def api_fire_alarm():
    """Activate or deactivate fire alarm."""
    data = request.get_json()
    action = data.get("action", "")

    if action == "activate":
        door_controller.trigger_fire_alarm()
        add_alert("FIRE", "Fire alarm activated! All doors unlocked for evacuation.")
        add_access_log(None, "SYSTEM", "-", "🔥 Emergency", "Emergency", "Opened")
        return jsonify({"success": True, "message": "Fire alarm activated!"})
    elif action == "deactivate":
        door_controller.deactivate_fire_alarm()
        resolve_all_alerts()
        add_access_log(None, "SYSTEM", "-", "✅ Resolved", "Emergency", "Locked")
        return jsonify({"success": True, "message": "Fire alarm deactivated."})

    return jsonify({"success": False, "message": "Invalid action"}), 400


@app.route("/api/status")
def api_status():
    """Return current system status."""
    status = door_controller.get_status()
    alerts = get_active_alerts()
    return jsonify({
        "door_state": status["door_state"],
        "fire_alarm_active": status["fire_alarm_active"],
        "active_alerts": len(alerts),
        "last_logs": status["last_logs"],
    })


@app.route("/api/model_info")
def api_model_info():
    """Return info about which AI models are being used."""
    info = get_model_info()
    return jsonify(info)


# ─── Helpers ─────────────────────────────────────────────────────

def _clear_deepface_cache():
    """Clear DeepFace representations cache so new faces are discovered."""
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.startswith("ds_model_") or f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            if f.startswith("representations_"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass


# ─── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔐 Smart Door Lock System Starting...")
    print("📡 Open http://localhost:5000 in your browser")
    print("👤 Default login: admin / admin123\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
