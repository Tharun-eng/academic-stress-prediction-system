import os

# IMPORTANT: Must come BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from flask import redirect, url_for
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_connection, save_prediction
import pandas as pd
from flask import send_file
from flask import flash
import joblib
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# ===================================
# Load Model & Encoders
# ===================================
model = joblib.load("models/stress_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

print("Gender:", label_encoders["Gender"].classes_)
print("Tuition:", label_encoders["Tuition"].classes_)
print("Physical_Exercise:", label_encoders["Physical_Exercise"].classes_)
print("Family_Income_Level:", label_encoders["Family_Income_Level"].classes_)
print("University_Type:", label_encoders["University_Type"].classes_)

# =====================================
# PRINT FEATURE IMPORTANCE (RUN ONCE)
# =====================================

try:
    feature_names = [
        "Age",
        "Gender",
        "Study_Hours",
        "Class_Attendance",
        "Tuition",
        "Exam_Frequency",
        "Assignment_Load",
        "Sleep_Hours",
        "Physical_Exercise",
        "Social_Media_Use",
        "Screen_Time",
        "Family_Income_Level",
        "Peer_Pressure",
        "Family_Support",
        "Anxiety_Level",
        "University_Type"
    ]

    print("\n========== FEATURE IMPORTANCE ==========\n")

    for feature, importance in zip(feature_names, model.feature_importances_):
        print(f"{feature:25} : {importance:.4f}")

    print("\n========================================\n")

except AttributeError:
    print("This model does not support feature importance.")

history_file = "history/prediction_history.csv"

os.makedirs("history", exist_ok=True)

os.makedirs("static/charts", exist_ok=True)


# ===================================
# Read Prediction History
# ===================================
# ===================================
# Read Prediction History
# ===================================
def load_history():

    connection = get_connection()

    cursor = connection.cursor(dictionary=True)

    query = """
    SELECT
        id,
        DateTime,
        Age,
        Study_Hours,
        Screen_Time,
        Sleep_Hours,
        Prediction
    FROM prediction_history
    ORDER BY id DESC
    LIMIT 10
    """

    cursor.execute(query)

    history = cursor.fetchall()
    

    cursor.close()
    connection.close()

    return history

# ===================================
# Dashboard Statistics
# ===================================

def get_dashboard_stats():

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    user_id = session["user_id"]

    # Total Predictions
    cursor.execute(
        "SELECT COUNT(*) AS total FROM prediction_history WHERE user_id=%s",
        (user_id,)
    )
    total = cursor.fetchone()["total"]

    # Low Stress
    cursor.execute(
        "SELECT COUNT(*) AS low_count FROM prediction_history WHERE Prediction='Low' AND user_id=%s",
        (user_id,)
    )
    low = cursor.fetchone()["low_count"]

    # Medium Stress
    cursor.execute(
        "SELECT COUNT(*) AS medium_count FROM prediction_history WHERE Prediction='Medium' AND user_id=%s",
        (user_id,)
    )
    medium = cursor.fetchone()["medium_count"]

    # High Stress
    cursor.execute(
        "SELECT COUNT(*) AS high_count FROM prediction_history WHERE Prediction='High' AND user_id=%s",
        (user_id,)
    )
    high = cursor.fetchone()["high_count"]

    # Average Values
    cursor.execute("""
        SELECT
            AVG(Study_Hours) AS avg_study,
            AVG(Screen_Time) AS avg_screen,
            AVG(Sleep_Hours) AS avg_sleep
        FROM prediction_history
        WHERE user_id=%s
    """, (user_id,))

    avg = cursor.fetchone()

    cursor.close()
    connection.close()

    return {

        "total": total or 0,

        "low": low or 0,

        "medium": medium or 0,

        "high": high or 0,

        "avg_study": round(avg["avg_study"] or 0, 1),

        "avg_screen": round(avg["avg_screen"] or 0, 1),

        "avg_sleep": round(avg["avg_sleep"] or 0, 1)

    }
# ===================================
# Home Page # Bar Chart
# ===================================
# ===================================
# Generate Bar Chart (MySQL)
# ===================================

def generate_bar_chart():

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT Prediction, COUNT(*) AS total
        FROM prediction_history
        WHERE user_id=%s
        GROUP BY Prediction
    """, (session["user_id"],))

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    # Default values
    low = 0
    medium = 0
    high = 0

    # Read values
    for row in rows:

        if row["Prediction"] == "Low":
            low = row["total"]

        elif row["Prediction"] == "Medium":
            medium = row["total"]

        elif row["Prediction"] == "High":
            high = row["total"]

    labels = ["Low", "Medium", "High"]
    values = [low, medium, high]
    colors = ["green", "orange", "red"]

    plt.figure(figsize=(7,5))

    bars = plt.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.2
    )

    # Display values above bars
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05,
            str(int(height)),
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    plt.title(
        "Your Stress Level Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Stress Level", fontsize=12)

    plt.ylabel("Number of Predictions", fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    import os

    chart_path = "static/charts/bar_chart.png"

    if os.path.exists(chart_path):
        os.remove(chart_path)

    plt.savefig(chart_path, dpi=200)

    plt.close("all")

    print("========== USER BAR CHART UPDATED ==========")
    print("User ID :", session["user_id"])
    print("Low     :", low)
    print("Medium  :", medium)
    print("High    :", high)
    print("============================================")

# ===================================
def generate_pie_chart():

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT Prediction, COUNT(*) AS total
        FROM prediction_history
        WHERE user_id=%s
        GROUP BY Prediction
    """, (session["user_id"],))

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    if not rows:
        return

    labels = []
    values = []
    colors = []

    for row in rows:

        labels.append(row["Prediction"])
        values.append(row["total"])

        if row["Prediction"] == "Low":
            colors.append("green")

        elif row["Prediction"] == "Medium":
            colors.append("gold")

        else:
            colors.append("red")

    plt.figure(figsize=(6, 6))

    plt.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Your Stress Level Distribution")

    plt.tight_layout()

    plt.savefig("static/charts/pie_chart.png", dpi=150)

    plt.close("all")

# Home Page
# ===================================
@app.route("/")
def home():

    # =====================================
    # User Authentication
    # =====================================

    if "user_id" not in session:

        flash(
            "Please login first.",
            "warning"
        )

        return redirect(url_for("login"))

    # =====================================
    # Generate Charts
    # =====================================

    generate_pie_chart()
    generate_bar_chart()

    # =====================================
    # Search & Filter
    # =====================================

    search = request.args.get("search", "").strip()
    stress_filter = request.args.get("filter", "All")

    # =====================================
    # Load Prediction History
    # =====================================

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            DateTime,
            Age,
            Study_Hours,
            Screen_Time,
            Sleep_Hours,
            Prediction
        FROM prediction_history
        WHERE user_id = %s
    """

    values = [session["user_id"]]

    # Search by Age

    if search:
        query += " AND Age = %s"
        values.append(search)

    if stress_filter != "All":
        query += " AND Prediction = %s"
        values.append(stress_filter)

    query += " ORDER BY id DESC LIMIT 10"

    cursor.execute(query, values)

    history = cursor.fetchall()

    cursor.close()
    connection.close()

    # =====================================
    # Dashboard Statistics
    # =====================================

    stats = get_dashboard_stats()

    # =====================================
    # Read Prediction Result
    # =====================================

    prediction = session.pop("prediction", None)

    recommendation = session.pop("recommendation", None)

    color = session.pop("color", None)

    icon = session.pop("icon", None)

    # =====================================
    # Render Home Page
    # =====================================

    return render_template(

        "index.html",

        # Logged-in User
        user_name=session.get("user_name"),

        # Dashboard
        stats=stats,

        # Prediction Result
        prediction=prediction,
        recommendation=recommendation,
        color=color,
        icon=icon,

        # History
        history=history,

        # Search & Filter
        search=search,
        filter=stress_filter

    )


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]

        university = request.form["university"]
        course = request.form["course"]
        academic_year = request.form["academic_year"]

        # Encrypt password
        hashed_password = generate_password_hash(password)

        connection = get_connection()
        cursor = connection.cursor()

        try:

            query = """
            INSERT INTO users
            (
                full_name,
                email,
                password,
                university,
                course,
                academic_year
            )
            VALUES
            (%s,%s,%s,%s,%s,%s)
            """

            values = (
                full_name,
                email,
                hashed_password,
                university,
                course,
                academic_year
            )

            cursor.execute(query, values)

            connection.commit()

            flash(
                "Registration Successful! Please login.",
                "success"
            )

            return redirect(url_for("login"))

        except Exception as e:

            flash(
                "Email already exists!",
                "danger"
            )

            print(e)

        finally:

            cursor.close()
            connection.close()

    return render_template("register.html")

# ===================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user and check_password_hash(
                user["password"],
                password
        ):

            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["user_email"] = user["email"]

            flash(
                f"Welcome {user['full_name']}!",
                "success"
            )

            return redirect(url_for("home"))

        else:

            flash(
                "Invalid Email or Password!",
                "danger"
            )

    return render_template("login.html")

#-----------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "Logged out successfully.",
        "success"
    )

    return redirect(url_for("login"))


# Complete Prediction History
# ===================================
@app.route("/history")
def history_page():

    if "user_id" not in session:

        flash(
            "Please login first.",
            "warning"
        )

        return redirect(url_for("login"))

    search = request.args.get("search", "").strip()
    stress_filter = request.args.get("filter", "All")

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            DateTime,
            Age,
            Study_Hours,
            Screen_Time,
            Sleep_Hours,
            Prediction
        FROM prediction_history
        WHERE user_id = %s
    """

    params = [session["user_id"]]

    # Search by Age
    if search != "":
        query += " AND Age = %s"
        params.append(int(search))

    # Filter by Stress Level
    if stress_filter != "All":
        query += " AND Prediction = %s"
        params.append(stress_filter)

    query += " ORDER BY id DESC"

    print("SQL Query :", query)
    print("Parameters :", params)

    print("=" * 50)
    print("QUERY:")
    print(query)
    print("PARAMS:")
    print(params)
    print("=" * 50)

    if len(params) == 0:
        cursor.execute(query)
    else:
        cursor.execute(query, params)

    history = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "history.html",
        history=history,
        search=search,
        filter=stress_filter
    )

# ===================================
# Delete Prediction (MySQL)
# ===================================

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_prediction(id):

    # -----------------------------
    # Login Required
    # -----------------------------

    if "user_id" not in session:

        flash("Please login first.", "warning")

        return redirect(url_for("login"))

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    # -----------------------------
    # Load only current user's record
    # -----------------------------

    cursor.execute("""

        SELECT *

        FROM prediction_history

        WHERE id=%s

        AND user_id=%s

    """, (id, session["user_id"]))

    row = cursor.fetchone()

    # -----------------------------
    # Security Check
    # -----------------------------

    if row is None:

        cursor.close()
        connection.close()

        flash(
            "Access denied. This prediction does not belong to your account.",
            "danger"
        )

        return redirect(url_for("history_page"))

    # =====================================================
    # UPDATE
    # =====================================================

    if request.method == "POST":

        age = int(request.form["Age"])
        gender = request.form["Gender"]

        study_hours = int(request.form["Study_Hours"])
        class_attendance = int(request.form["Class_Attendance"])
        tuition = request.form["Tuition"]

        exam_frequency = int(request.form["Exam_Frequency"])
        assignment_load = int(request.form["Assignment_Load"])

        sleep_hours = int(request.form["Sleep_Hours"])
        social_media_use = int(request.form["Social_Media_Use"])
        screen_time = int(request.form["Screen_Time"])

        physical_exercise = request.form["Physical_Exercise"]
        family_income = request.form["Family_Income_Level"]

        peer_pressure = int(request.form["Peer_Pressure"])
        family_support = int(request.form["Family_Support"])
        anxiety_level = int(request.form["Anxiety_Level"])

        university_type = request.form["University_Type"]

        input_data = pd.DataFrame([{

            "Age": age,

            "Gender":
            label_encoders["Gender"].transform([gender])[0],

            "Study_Hours": study_hours,

            "Class_Attendance": class_attendance,

            "Tuition":
            label_encoders["Tuition"].transform([tuition])[0],

            "Exam_Frequency": exam_frequency,

            "Assignment_Load": assignment_load,

            "Sleep_Hours": sleep_hours,

            "Physical_Exercise":
            label_encoders["Physical_Exercise"].transform(
                [physical_exercise]
            )[0],

            "Social_Media_Use": social_media_use,

            "Screen_Time": screen_time,

            "Family_Income_Level":
            label_encoders["Family_Income_Level"].transform(
                [family_income]
            )[0],

            "Peer_Pressure": peer_pressure,

            "Family_Support": family_support,

            "Anxiety_Level": anxiety_level,

            "University_Type":
            label_encoders["University_Type"].transform(
                [university_type]
            )[0]

        }])

        pred = model.predict(input_data)[0]

        prediction = label_encoders["Stress_Level"].inverse_transform(
            [pred]
        )[0]

        # -----------------------------
        # Update only current user's row
        # -----------------------------

        cursor.execute("""

            UPDATE prediction_history

            SET

                Age=%s,
                Gender=%s,
                Study_Hours=%s,
                Class_Attendance=%s,
                Tuition=%s,
                Exam_Frequency=%s,
                Assignment_Load=%s,
                Sleep_Hours=%s,
                Physical_Exercise=%s,
                Social_Media_Use=%s,
                Screen_Time=%s,
                Family_Income_Level=%s,
                Peer_Pressure=%s,
                Family_Support=%s,
                Anxiety_Level=%s,
                University_Type=%s,
                Prediction=%s

            WHERE id=%s

            AND user_id=%s

        """, (

            age,
            gender,
            study_hours,
            class_attendance,
            tuition,
            exam_frequency,
            assignment_load,
            sleep_hours,
            physical_exercise,
            social_media_use,
            screen_time,
            family_income,
            peer_pressure,
            family_support,
            anxiety_level,
            university_type,
            prediction,
            id,
            session["user_id"]

        ))

        connection.commit()

        cursor.close()
        connection.close()

        flash("Prediction updated successfully!", "success")

        return redirect(url_for("history_page"))

    cursor.close()
    connection.close()

    return render_template(
        "edit_prediction.html",
        row=row
    )

from flask import flash

# ==========================================
# Delete Prediction
# ==========================================

@app.route("/delete/<int:id>")
def delete_prediction(id):

    # User must be logged in
    if "user_id" not in session:

        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    connection = get_connection()
    cursor = connection.cursor()

    # Delete only the logged-in user's record
    cursor.execute(
        """
        DELETE FROM prediction_history
        WHERE id=%s
        AND user_id=%s
        """,
        (
            id,
            session["user_id"]
        )
    )

    connection.commit()

    if cursor.rowcount > 0:

        flash(
            "Prediction deleted successfully!",
            "success"
        )

    else:

        flash(
            "Prediction not found or permission denied.",
            "danger"
        )

    cursor.close()
    connection.close()

    return redirect(url_for("history_page"))




@app.route("/export_csv")
def export_csv():

    if "user_id" not in session:

        flash(
            "Please login first.",
            "warning"
        )

        return redirect(url_for("login"))

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            DateTime,
            Age,
            Study_Hours,
            Screen_Time,
            Sleep_Hours,
            Prediction
        FROM prediction_history
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    cursor.close()
    connection.close()

    df = pd.DataFrame(data)

    filename = "Prediction_History.csv"

    df.to_csv(filename, index=False)

    return send_file(
        filename,
        as_attachment=True,
        download_name="Prediction_History.csv"
    )

#----------------------------------------
@app.route("/export_current_csv")
def export_current_csv():

    if "current_prediction" not in session:

        return redirect(url_for("home"))

    df = pd.DataFrame([session["current_prediction"]])

    filename = "Current_Prediction.csv"

    df.to_csv(filename, index=False)

    return send_file(

        filename,

        as_attachment=True

    )

#-----------------------------------------

@app.route("/export_current_pdf")
def export_current_pdf():

    if "current_prediction" not in session:

        return redirect(url_for("home"))

    data = session["current_prediction"]

    pdf_file = "Current_Prediction_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(

        "<b><font size=18>Current Prediction Report</font></b>",

        styles["Title"]

    )

    elements.append(title)

    elements.append(

        Paragraph(

            "<br/>",

            styles["Normal"]

        )

    )

    table_data = [

        ["Field", "Value"],

        ["Date & Time", data["DateTime"]],

        ["Age", data["Age"]],

        ["Study Hours", data["Study_Hours"]],

        ["Screen Time", data["Screen_Time"]],

        ["Sleep Hours", data["Sleep_Hours"]],

        ["Stress Level", data["Prediction"]]

    ]

    table = Table(table_data)

    table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.darkblue),

        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("GRID",(0,0),(-1,-1),1,colors.black),

        ("BACKGROUND",(0,1),(-1,-1),colors.beige),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")

    ]))

    elements.append(table)

    doc.build(elements)

    return send_file(

        pdf_file,

        as_attachment=True

    )


#----------------------------------------
@app.route("/export_pdf")
def export_pdf():

    if "user_id" not in session:

        flash(
            "Please login first.",
            "warning"
        )

        return redirect(url_for("login"))

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            DATE_FORMAT(DateTime,'%d-%m-%Y %H:%i') AS DateTime,
            Age,
            Study_Hours,
            Screen_Time,
            Sleep_Hours,
            Prediction
        FROM prediction_history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    cursor.close()
    connection.close()

    if len(records) == 0:
        return redirect(url_for("history_page"))

    pdf_file = "Prediction_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    # ==========================
    # Title
    # ==========================
    title = Paragraph(
        "<b><font size=18>Academic Stress Prediction Report</font></b>",
        styles["Title"]
    )

    elements.append(title)

    elements.append(
        Paragraph(
            f"Generated on : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    # ==========================
    # Summary
    # ==========================

    low = sum(1 for row in records if row["Prediction"] == "Low")
    medium = sum(1 for row in records if row["Prediction"] == "Medium")
    high = sum(1 for row in records if row["Prediction"] == "High")

    summary = [
        ["Category", "Count"],
        ["Total Predictions", len(records)],
        ["Low Stress", low],
        ["Medium Stress", medium],
        ["High Stress", high]
    ]

    summary_table = Table(summary)

    summary_table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),

        ("GRID", (0,0), (-1,-1), 1, colors.black),

        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),

        ("ALIGN", (0,0), (-1,-1), "CENTER"),

        ("BOTTOMPADDING", (0,0), (-1,-1), 8)

    ]))

    elements.append(summary_table)

    elements.append(Paragraph("<br/><br/>", styles["Normal"]))

    elements.append(
        Paragraph("Prediction History", styles["Heading2"])
    )

    elements.append(Paragraph("<br/>", styles["Normal"]))

    # ==========================
    # Prediction Table
    # ==========================

    data = [[
        "DateTime",
        "Age",
        "Study",
        "Screen",
        "Sleep",
        "Prediction"
    ]]

    for row in records:

        data.append([

            row["DateTime"],

            row["Age"],

            row["Study_Hours"],

            row["Screen_Time"],

            row["Sleep_Hours"],

            row["Prediction"]

        ])

    table = Table(data)

    table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),

        ("TEXTCOLOR", (0,0), (-1,0), colors.white),

        ("GRID", (0,0), (-1,-1), 1, colors.black),

        ("BACKGROUND", (0,1), (-1,-1), colors.beige),

        ("ALIGN", (0,0), (-1,-1), "CENTER"),

        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

        ("BOTTOMPADDING", (0,0), (-1,0), 10)

    ]))

    elements.append(table)

    # ==========================
    # Build PDF
    # ==========================

    doc.build(elements)

    return send_file(
        pdf_file,
        as_attachment=True
    )
# Prediction
# ===================================
@app.route("/predict", methods=["POST"])
def predict():

    print("Logged in User ID:", session.get("user_id"))

    age = int(request.form["age"])
    gender = request.form["gender"]

    study_hours = int(request.form["study_hours"])
    class_attendance = int(request.form["class_attendance"])
    tuition = request.form["tuition"]
    exam_frequency = int(request.form["exam_frequency"])
    assignment_load = int(request.form["assignment_load"])

    sleep_hours = int(request.form["sleep_hours"])
    social_media_use = int(request.form["social_media_use"])
    screen_time = int(request.form["screen_time"])

    physical_exercise = request.form["physical_exercise"]
    family_income = request.form["family_income"]
    peer_pressure = int(request.form["peer_pressure"])
    family_support = int(request.form["family_support"])
    anxiety_level = int(request.form["anxiety_level"])
    university_type = request.form["university_type"]

    input_data = {

        "Age": int(request.form["age"]),

        "Gender":
        label_encoders["Gender"].transform([gender])[0],

        "Study_Hours":
        int(request.form["study_hours"]),

        "Class_Attendance":
        int(request.form["class_attendance"]),

        "Tuition":
        label_encoders["Tuition"].transform([tuition])[0],

        "Exam_Frequency":
        int(request.form["exam_frequency"]),

        "Assignment_Load":
        int(request.form["assignment_load"]),

        "Sleep_Hours":
        int(request.form["sleep_hours"]),

        "Physical_Exercise":
        label_encoders["Physical_Exercise"].transform(
            [physical_exercise]
        )[0],

        "Social_Media_Use":
        int(request.form["social_media_use"]),

        "Screen_Time":
        int(request.form["screen_time"]),

        "Family_Income_Level":
        label_encoders["Family_Income_Level"].transform(
            [family_income]
        )[0],

        "Peer_Pressure":
        int(request.form["peer_pressure"]),

        "Family_Support":
        int(request.form["family_support"]),

        "Anxiety_Level":
        int(request.form["anxiety_level"]),

        "University_Type":
        label_encoders["University_Type"].transform(
            [university_type]
        )[0]

    }

    # ------------------------
    # Prediction
    # ------------------------

    # ------------------------
# Prediction
# ------------------------

    df = pd.DataFrame([input_data])

    print("\n=========== INPUT TO MODEL ===========")
    print(df)
    print("======================================")

    pred = model.predict(df)[0]

    prediction = label_encoders["Stress_Level"].inverse_transform([pred])[0]

    print("\n=========== MODEL OUTPUT ===========")
    print("Encoded Prediction :", pred)
    print("Stress Level :", prediction)
    print("====================================")

    # ------------------------
    # Recommendation
    # ------------------------

    if prediction == "Low":

        color = "success"

        icon = "🟢"

        recommendation = """
        • Maintain your healthy routine.<br>
        • Continue getting adequate sleep.<br>
        • Keep balancing study and leisure activities.
        """

    elif prediction == "Medium":

        color = "warning"

        icon = "🟡"

        recommendation = """
        • Reduce screen time.<br>
        • Take regular study breaks.<br>
        • Practice time management.<br>
        • Sleep at least 7 hours daily.
        """

    else:

        color = "danger"

        icon = "🔴"

        recommendation = """
        • High academic stress detected.<br>
        • Reduce screen time.<br>
        • Improve your sleep schedule.<br>
        • Consult a mentor or counselor.
        """
    # ------------------------
# Store Prediction in Session
# ------------------------

    session["prediction"] = prediction
    session["color"] = color
    session["icon"] = icon
    session["recommendation"] = recommendation
    # ------------------------
    # Save Prediction
    # ------------------------

    new_record = pd.DataFrame([{

        "DateTime":
        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

        "Age":
        input_data["Age"],

        "Study_Hours":
        input_data["Study_Hours"],

        "Screen_Time":
        input_data["Screen_Time"],

        "Sleep_Hours":
        input_data["Sleep_Hours"],

        "Prediction":
        prediction

    }])
    
    # Save current prediction in session

    session["current_prediction"] = {

    "DateTime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

    "Age": input_data["Age"],

    "Study_Hours": input_data["Study_Hours"],

    "Screen_Time": input_data["Screen_Time"],

    "Sleep_Hours": input_data["Sleep_Hours"],

    "Prediction": prediction

}
    # ------------------------
# Save Prediction to MySQL
# ------------------------

    # ------------------------
# Save Prediction to MySQL
# ------------------------

    connection = get_connection()
    cursor = connection.cursor()

    query = """
    INSERT INTO prediction_history
    (
        DateTime,
        Age,
        Study_Hours,
        Screen_Time,
        Sleep_Hours,
        Prediction,
        Gender,
        Class_Attendance,
        Exam_Frequency,
        Assignment_Load,
        Social_Media_Use,
        Peer_Pressure,
        Family_Support,
        Anxiety_Level,
        Tuition,
        Physical_Exercise,
        Family_Income_Level,
        University_Type,
        user_id
    )
    VALUES
    (
        %s,%s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,%s,
        %s
    )
    """

    values = (
        datetime.now(),
        age,
        study_hours,
        screen_time,
        sleep_hours,
        prediction,
        gender,
        class_attendance,
        exam_frequency,
        assignment_load,
        social_media_use,
        peer_pressure,
        family_support,
        anxiety_level,
        tuition,
        physical_exercise,
        family_income,
        university_type,
        session["user_id"]
    )

    cursor.execute(query, values)

    connection.commit()

    cursor.close()
    connection.close()

    # ------------------------
    # Redirect to Home
    # ------------------------

    return redirect(url_for("home"))


# ===================================



@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json()

        input_data = {

            "Age": data["age"],

            "Gender": label_encoders["Gender"].transform(
                [data["gender"]]
            )[0],

            "Study_Hours": data["studyHours"],

            "Class_Attendance": data["classAttendance"],

            "Tuition": label_encoders["Tuition"].transform(
                [data["tuition"]]
            )[0],

            "Exam_Frequency": data["examFrequency"],

            "Assignment_Load": data["assignmentLoad"],

            "Sleep_Hours": data["sleepHours"],

            "Physical_Exercise": label_encoders["Physical_Exercise"].transform(
                [data["physicalExercise"]]
            )[0],

            "Social_Media_Use": data["socialMediaUse"],

            "Screen_Time": data["screenTime"],

            "Family_Income_Level": label_encoders["Family_Income_Level"].transform(
                [data["familyIncome"]]
            )[0],

            "Peer_Pressure": data["peerPressure"],

            "Family_Support": data["familySupport"],

            "Anxiety_Level": data["anxietyLevel"],

            "University_Type": label_encoders["University_Type"].transform(
                [data["universityType"]]
            )[0]

        }

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        stress = label_encoders["Stress_Level"].inverse_transform(
            [prediction]
        )[0]

        if stress == "Low":

            recommendation = "Maintain your healthy routine."

            color = "green"

            icon = "🟢"

        elif stress == "Medium":

            recommendation = "Reduce screen time and take regular breaks."

            color = "orange"

            icon = "🟡"

        else:

            recommendation = "High stress detected. Please consult your mentor."

            color = "red"

            icon = "🔴"

        return jsonify({

            "prediction": stress,

            "recommendation": recommendation,

            "color": color,

            "icon": icon

        })

    except Exception as e:
        print("\n========== API ERROR ==========")
        import traceback
        traceback.print_exc()
        print("===============================\n")

        return jsonify({
            "error": str(e)
        }), 500


# Run Flask
# ===================================

if __name__ == "__main__":
    from waitress import serve

    serve(
        app,
        host="0.0.0.0",
        port=5000
    )