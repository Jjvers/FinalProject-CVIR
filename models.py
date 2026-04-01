"""
Database Models for Smart Door Lock System
Uses SQLite for lightweight local storage.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartdoor.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    cursor = conn.cursor()

    # Classes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL UNIQUE,
            description TEXT,
            start_time TEXT DEFAULT '',
            end_time TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            face_image_path TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes(id)
        )
    """)

    # Access Logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            student_name TEXT,
            class_name TEXT,
            detected_mood TEXT,
            status TEXT NOT NULL,
            door_action TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    # Alerts table (Fire, Emergency, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            message TEXT,
            is_active INTEGER DEFAULT 1,
            triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        )
    """)

    # Admin users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    # Insert default admin if not exists
    cursor.execute("SELECT COUNT(*) FROM admins")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO admins (username, password) VALUES (?, ?)",
            ("admin", "admin123"),
        )

    conn.commit()
    conn.close()


# ─── Class Operations ───────────────────────────────────────────

def add_class(class_name, description="", start_time="", end_time=""):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO classes (class_name, description, start_time, end_time) VALUES (?, ?, ?, ?)",
            (class_name, description, start_time, end_time),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_all_classes():
    conn = get_db()
    classes = conn.execute("SELECT * FROM classes ORDER BY class_name").fetchall()
    conn.close()
    return classes


def delete_class(class_id):
    conn = get_db()
    conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()


def get_class_by_id(class_id):
    conn = get_db()
    cls = conn.execute("SELECT * FROM classes WHERE id = ?", (class_id,)).fetchone()
    conn.close()
    return cls


# ─── Student Operations ─────────────────────────────────────────

def add_student(student_id, name, class_id, face_image_path):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO students (student_id, name, class_id, face_image_path) VALUES (?, ?, ?, ?)",
            (student_id, name, class_id, face_image_path),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_all_students():
    conn = get_db()
    students = conn.execute("""
        SELECT s.*, c.class_name
        FROM students s
        JOIN classes c ON s.class_id = c.id
        ORDER BY s.name
    """).fetchall()
    conn.close()
    return students


def get_student_by_id(student_id):
    conn = get_db()
    student = conn.execute("""
        SELECT s.*, c.class_name
        FROM students s
        JOIN classes c ON s.class_id = c.id
        WHERE s.id = ?
    """, (student_id,)).fetchone()
    conn.close()
    return student


def get_students_by_class(class_id):
    conn = get_db()
    students = conn.execute("""
        SELECT s.*, c.class_name
        FROM students s
        JOIN classes c ON s.class_id = c.id
        WHERE s.class_id = ?
        ORDER BY s.name
    """, (class_id,)).fetchall()
    conn.close()
    return students


def delete_student(student_db_id):
    conn = get_db()
    student = conn.execute("SELECT face_image_path FROM students WHERE id = ?", (student_db_id,)).fetchone()
    if student and student["face_image_path"] and os.path.exists(student["face_image_path"]):
        os.remove(student["face_image_path"])
    conn.execute("DELETE FROM students WHERE id = ?", (student_db_id,))
    conn.commit()
    conn.close()


# ─── Access Log Operations ──────────────────────────────────────

def add_access_log(student_id, student_name, class_name, detected_mood, status, door_action):
    conn = get_db()
    conn.execute(
        """INSERT INTO access_logs
           (student_id, student_name, class_name, detected_mood, status, door_action)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (student_id, student_name, class_name, detected_mood, status, door_action),
    )
    conn.commit()
    conn.close()


def get_access_logs(limit=50):
    conn = get_db()
    logs = conn.execute(
        "SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return logs


# ─── Alert Operations ───────────────────────────────────────────

def add_alert(alert_type, message):
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts (alert_type, message) VALUES (?, ?)",
        (alert_type, message),
    )
    conn.commit()
    conn.close()


def get_active_alerts():
    conn = get_db()
    alerts = conn.execute(
        "SELECT * FROM alerts WHERE is_active = 1 ORDER BY triggered_at DESC"
    ).fetchall()
    conn.close()
    return alerts


def resolve_alert(alert_id):
    conn = get_db()
    conn.execute(
        "UPDATE alerts SET is_active = 0, resolved_at = ? WHERE id = ?",
        (datetime.now().isoformat(), alert_id),
    )
    conn.commit()
    conn.close()


def resolve_all_alerts():
    conn = get_db()
    conn.execute(
        "UPDATE alerts SET is_active = 0, resolved_at = ?",
        (datetime.now().isoformat(),),
    )
    conn.commit()
    conn.close()


# ─── Admin Operations ───────────────────────────────────────────

def verify_admin(username, password):
    conn = get_db()
    admin = conn.execute(
        "SELECT * FROM admins WHERE username = ? AND password = ?",
        (username, password),
    ).fetchone()
    conn.close()
    return admin is not None
