from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
from datetime import datetime
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

# ReportLab Imports for PDF Generation
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# Machine Learning Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
app.secret_key = "sickbay_ai_secret"

# =============================
# LOAD AI MODEL
# =============================
try:
    model = joblib.load("sickbay_ai_model.pkl")
    vectorizer = joblib.load("symptom_vectorizer.pkl")
except Exception:
    model = None
    vectorizer = None


# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def connect_db():
    conn = sqlite3.connect("sickbay.db")
    conn.row_factory = sqlite3.Row
    return conn


# =============================
# ROLE CHECKER
# =============================
def check_role(role):
    if "user" not in session:
        return False
    return session.get("role") == role


# =============================
# CREATE DATABASE TABLES & SEED DATA
# =============================
def create_database():
    db = connect_db()

    # USERS TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # STUDENTS TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE,
        name TEXT,
        age TEXT,
        gender TEXT,
        class_name TEXT,
        parent_phone TEXT,
        nhis_number TEXT,
        nhis_status TEXT
    )
    """)

    # MEDICINE TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS medicines(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        medicine_name TEXT,
        quantity TEXT,
        expiry_date TEXT,
        supplier TEXT
    )
    """)

    # MEDICAL RECORDS TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS medical_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        symptoms TEXT,
        diagnosis TEXT,
        treatment TEXT,
        nurse TEXT,
        date TEXT
    )
    """)

    # NURSE NOTES TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS nurse_notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        note TEXT,
        date TEXT
    )
    """)

    # AI TRAINING DATA TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS ai_training_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptoms TEXT,
        diagnosis TEXT,
        treatment TEXT,
        nurse TEXT,
        date TEXT
    )
    """)

    # AI TRAINING HISTORY TABLE
    db.execute("""
    CREATE TABLE IF NOT EXISTS ai_training_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trained_by TEXT,
        training_date TEXT,
        data_count INTEGER,
        status TEXT
    )
    """)

    # DEFAULT HASHED USERS
    default_nurse_pass = generate_password_hash("1234")
    default_admin_pass = generate_password_hash("1234")

    db.execute("""
    INSERT OR IGNORE INTO users (username, password, role)
    VALUES ('nurse', ?, 'Nurse')
    """, (default_nurse_pass,))

    db.execute("""
    INSERT OR IGNORE INTO users (username, password, role)
    VALUES ('admin', ?, 'Admin')
    """, (default_admin_pass,))

    # DEFAULT AI TRAINING DATA SAMPLES
    training_samples = [
        ("fever, severe headache, joint pain, chills, fatigue", "Malaria", "Paracetamol, Artemether-Lumefantrine", "nurse"),
        ("high temperature, cold shivers, loss of appetite, body ache", "Malaria", "Antimalarial medication, Bed rest", "nurse"),
        ("stomach pain, vomiting, watery diarrhea, nausea", "Food Poisoning", "ORSalts, Antiemetic, Hydration", "nurse"),
        ("abdominal cramps, nausea, loose stools after lunch", "Food Poisoning", "Oral Rehydration Salts, Light meal", "nurse"),
        ("throbbing head pain, sensitivity to light, stress, eye strain", "Migraine / Tension Headache", "Paracetamol, Dark room rest", "nurse"),
        ("mild headache, fatigue, heavy studying", "Tension Headache", "Rest, Hydration, Paracetamol", "nurse"),
        ("runny nose, sneezing, mild cough, sore throat", "Common Cold", "Cough syrup, Warm water, Vitamin C", "nurse"),
        ("nasal congestion, sneezing, watery eyes, fever", "Common Cold", "Decongestant, Vitamin C, Rest", "nurse"),
        ("wheezing, shortness of breath, chest tightness, coughing", "Asthma Flare-up", "Salbutamol Inhaler, Rest", "nurse"),
        ("difficulty breathing, tight chest after exercise", "Exercise-Induced Asthma", "Inhaler, Rest in cool room", "nurse"),
        ("painful swallowing, swollen tonsils, fever, sore throat", "Tonsillitis", "Antibiotics (if prescribed), Warm salt gargle", "nurse"),
        ("dizziness, pale skin, lightheadedness, standing up quickly", "Dehydration / Low Blood Pressure", "Glucose drink, Water, Rest", "nurse"),
        ("scratch, bleeding knee, minor cut on hand", "Minor Injury", "Clean wound, Apply Antiseptic, Bandage", "nurse")
    ]

    for symptoms, diagnosis, treatment, nurse in training_samples:
        db.execute("""
            INSERT OR IGNORE INTO ai_training_data (symptoms, diagnosis, treatment, nurse, date)
            VALUES (?, ?, ?, ?, ?)
        """, (symptoms, diagnosis, treatment, nurse, datetime.now().strftime("%d/%m/%Y")))

    db.commit()
    db.close()


# Ensure database and tables are ready on launch
create_database()


# =============================
# LOGIN & LOGOUT (SECURED WITH HASH CHECKING)
# =============================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = connect_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        db.close()

        # Check hash match
        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]
            return redirect("/")

        # Generic error message for enhanced security
        error = "Authentication failed. Invalid login credentials."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    db = connect_db()

    students = db.execute("SELECT * FROM students").fetchall()
    student_count = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    record_count = db.execute("SELECT COUNT(*) FROM medical_records").fetchone()[0]
    medicine_count = db.execute("SELECT COUNT(*) FROM medicines").fetchone()[0]

    db.close()

    return render_template(
        "dashboard.html",
        students=students,
        user=session["user"],
        role=session["role"],
        student_count=student_count,
        record_count=record_count,
        medicine_count=medicine_count
    )


# -----------------------------
# REGISTER STUDENT
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        db = connect_db()
        db.execute("""
            INSERT INTO students
            (student_id, name, age, gender, class_name, parent_phone, nhis_number, nhis_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.form["student_id"],
            request.form["name"],
            request.form["age"],
            request.form["gender"],
            request.form["class_name"],
            request.form["phone"],
            request.form["nhis_number"],
            request.form["nhis_status"]
        ))
        db.commit()
        db.close()

        return redirect("/")

    return render_template("register_student.html")


# =============================
# AI SYMPTOM CHECKER
# =============================
@app.route("/symptoms", methods=["GET", "POST"])
def symptoms():
    if "user" not in session:
        return redirect("/login")

    result = "Enter symptoms to analyze"

    if request.method == "POST":
        user_input = request.form["symptoms"]

        if model is None or vectorizer is None:
            result = "AI model not trained yet. Ask Admin to train it."
        else:
            data = vectorizer.transform([user_input])
            prediction = model.predict(data)[0]
            confidence = model.predict_proba(data)
            score = int(max(confidence[0]) * 100)

            result = f"""🤖 AI Prediction

Possible Condition:
{prediction}

Confidence:
{score}%

Please consult the nurse for confirmation."""

    return render_template("symptoms.html", result=result)


# -----------------------------
# MEDICINES
# -----------------------------
@app.route("/medicines", methods=["GET", "POST"])
def medicines():
    if "user" not in session:
        return redirect("/login")

    db = connect_db()

    if request.method == "POST":
        db.execute("""
            INSERT INTO medicines (medicine_name, quantity, expiry_date, supplier)
            VALUES (?, ?, ?, ?)
        """, (
            request.form["medicine_name"],
            request.form["quantity"],
            request.form["expiry_date"],
            request.form["supplier"]
        ))
        db.commit()

    medicines = db.execute("SELECT * FROM medicines").fetchall()
    db.close()

    return render_template("medicines.html", medicines=medicines)


# -----------------------------
# MEDICAL RECORDS
# -----------------------------
@app.route("/records", methods=["GET", "POST"])
def records():
    if "user" not in session:
        return redirect("/login")

    db = connect_db()

    if request.method == "POST":
        db.execute("""
            INSERT INTO medical_records (student_id, symptoms, diagnosis, treatment, nurse, date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request.form["student_id"],
            request.form["symptoms"],
            request.form["diagnosis"],
            request.form["treatment"],
            session["user"],
            datetime.now().strftime("%d/%m/%Y")
        ))
        db.commit()

    records = db.execute("SELECT * FROM medical_records ORDER BY id DESC").fetchall()
    db.close()

    return render_template("records.html", records=records)


# -----------------------------
# SEARCH
# -----------------------------
@app.route("/search", methods=["GET", "POST"])
def search():
    if "user" not in session:
        return redirect("/login")

    results = []
    search_type = ""

    if request.method == "POST":
        query = request.form["query"]
        search_type = request.form["type"]

        db = connect_db()

        if search_type == "student":
            results = db.execute("""
                SELECT * FROM students
                WHERE student_id LIKE ? OR name LIKE ?
            """, ("%" + query + "%", "%" + query + "%")).fetchall()

        elif search_type == "medicine":
            results = db.execute("""
                SELECT * FROM medicines
                WHERE medicine_name LIKE ?
            """, ("%" + query + "%",)).fetchall()

        db.close()

    return render_template("search.html", results=results, search_type=search_type)


# -----------------------------
# STUDENT PROFILE
# -----------------------------
@app.route("/student/<student_id>")
def student_profile(student_id):
    if "user" not in session:
        return redirect("/login")

    db = connect_db()
    student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
    records = db.execute("SELECT * FROM medical_records WHERE student_id=? ORDER BY id DESC", (student_id,)).fetchall()
    db.close()

    return render_template("student_profile.html", student=student, records=records)


# =============================
# ADMIN DASHBOARD
# =============================
@app.route("/admin")
def admin():
    if not check_role("Admin"):
        return "Access Denied"

    db = connect_db()
    user_count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    student_count = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    db.close()

    return render_template("admin.html", user_count=user_count, student_count=student_count)


# =============================
# USER MANAGEMENT & PASSWORD CHANGE (HASHED)
# =============================
@app.route("/users")
def users():
    if not check_role("Admin"):
        return "Access Denied"

    db = connect_db()
    users = db.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    db.close()

    return render_template("users.html", users=users)


@app.route("/add_user", methods=["GET", "POST"])
def add_user():
    if not check_role("Admin"):
        return "Access Denied"

    if request.method == "POST":
        hashed_password = generate_password_hash(request.form["password"])

        db = connect_db()
        db.execute("""
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
        """, (
            request.form["username"],
            hashed_password,
            request.form["role"]
        ))
        db.commit()
        db.close()
        return redirect("/users")

    return render_template("add_user.html")


@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    if not check_role("Admin"):
        return "Access Denied"

    db = connect_db()

    if request.method == "POST":
        user_id = request.form["user_id"]
        new_hashed_password = generate_password_hash(request.form["new_password"])

        db.execute("""
            UPDATE users
            SET password=?
            WHERE id=?
        """, (new_hashed_password, user_id))

        db.commit()
        db.close()

        return redirect("/users")

    users = db.execute("SELECT id, username, role FROM users").fetchall()
    db.close()

    return render_template("change_password.html", users=users)


# =============================
# AI TRAINING CENTER
# =============================
@app.route("/ai_training")
def ai_training():
    if not check_role("Admin"):
        return "Access Denied"

    db = connect_db()
    training_count = db.execute("SELECT COUNT(*) FROM ai_training_data").fetchone()[0]
    history = db.execute("SELECT * FROM ai_training_history ORDER BY id DESC LIMIT 1").fetchone()
    db.close()

    return render_template("ai_training.html", training_count=training_count, history=history)


# =============================
# TRAIN AI MODEL
# =============================
@app.route("/train_ai", methods=["POST"])
def train_ai():
    global model, vectorizer

    if not check_role("Admin"):
        return "Access Denied"

    db = connect_db()
    data = db.execute("SELECT symptoms, diagnosis FROM ai_training_data WHERE diagnosis IS NOT NULL").fetchall()
    db.close()

    if len(data) < 2:
        return "Not enough training data"

    symptoms = [row["symptoms"] for row in data]
    diagnoses = [row["diagnosis"] for row in data]

    new_vectorizer = TfidfVectorizer()
    X = new_vectorizer.fit_transform(symptoms)

    new_model = MultinomialNB()
    new_model.fit(X, diagnoses)

    # Save trained models to disk
    joblib.dump(new_model, "sickbay_ai_model.pkl")
    joblib.dump(new_vectorizer, "symptom_vectorizer.pkl")

    # Update global variables in memory
    model = new_model
    vectorizer = new_vectorizer

    db = connect_db()
    db.execute("""
        INSERT INTO ai_training_history (trained_by, training_date, data_count, status)
        VALUES (?, ?, ?, ?)
    """, (
        session["user"],
        datetime.now().strftime("%d/%m/%Y %H:%M"),
        len(data),
        "Completed"
    ))

    db.commit()
    db.close()

    return redirect("/ai_training")


# =============================
# STUDENTS MANAGEMENT
# =============================
@app.route("/students")
def students():
    if "user" not in session:
        return redirect("/login")

    db = connect_db()
    students = db.execute("SELECT * FROM students").fetchall()
    db.close()

    return render_template("students.html", students=students)


@app.route("/delete_student/<student_id>")
def delete_student(student_id):
    if "user" not in session:
        return redirect("/login")

    db = connect_db()
    db.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    db.commit()
    db.close()

    return redirect("/students")


@app.route("/edit_student/<student_id>", methods=["GET", "POST"])
def edit_student(student_id):
    if "user" not in session:
        return redirect("/login")

    db = connect_db()

    if request.method == "POST":
        db.execute("""
            UPDATE students
            SET name=?, age=?, gender=?, class_name=?, parent_phone=?, nhis_number=?
            WHERE student_id=?
        """, (
            request.form["name"],
            request.form["age"],
            request.form["gender"],
            request.form["class_name"],
            request.form["phone"],
            request.form["nhis_number"],
            student_id
        ))
        db.commit()
        db.close()
        return redirect("/students")

    student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
    db.close()

    return render_template("edit_student.html", student=student)


# =============================
# GENERATE STUDENT PDF REPORT
# =============================
@app.route("/student_report/<student_id>")
def student_report(student_id):
    if "user" not in session:
        return redirect("/login")

    db = connect_db()
    student = db.execute("SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
    records = db.execute("SELECT * FROM medical_records WHERE student_id=? ORDER BY id DESC", (student_id,)).fetchall()
    db.close()

    if not student:
        return "Student not found", 404

    filename = f"{student_id}_Sickbay_AI_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    # HEADER
    content.append(Paragraph("🏥 SICKBAY AI", styles["Title"]))
    content.append(Paragraph("School Health Management System", styles["Heading3"]))
    content.append(Spacer(1, 20))

    # STUDENT INFORMATION
    student_data = [
        ["Student ID", student["student_id"]],
        ["Name", student["name"]],
        ["Class", student["class_name"]],
        ["Age", student["age"]],
        ["Gender", student["gender"]],
        ["NHIS Number", student["nhis_number"]],
        ["NHIS Status", student["nhis_status"]]
    ]

    table = Table(student_data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey)
    ]))

    content.append(Paragraph("👨‍🎓 Student Information", styles["Heading2"]))
    content.append(table)
    content.append(Spacer(1, 25))

    # MEDICAL HISTORY
    content.append(Paragraph("📋 Medical History", styles["Heading2"]))

    history = [["Date", "Symptoms", "Diagnosis", "Treatment"]]
    for record in records:
        history.append([
            record["date"],
            record["symptoms"],
            record["diagnosis"],
            record["treatment"]
        ])

    history_table = Table(history)
    history_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
    ]))

    content.append(history_table)
    content.append(Spacer(1, 30))

    # FOOTER
    content.append(Paragraph(f"""
        Report Generated: {datetime.now().strftime("%d/%m/%Y")}<br/><br/>
        Nurse: {session["user"]}<br/><br/>
        _______________________<br/>
        Nurse Signature
    """, styles["Normal"]))

    doc.build(content)

    return send_file(filename, as_attachment=True)


# -----------------------------
# START APP (LOCAL DEV SERVER)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
