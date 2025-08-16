# app.py
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
import pandas as pd
from Ecg import ECG
from roboflow import Roboflow
import supervision as sv

# ------------------- Flask Config -------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
PROCESS_FOLDER = 'process'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecg_app.db'  # SQLite DB file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ------------------- Database Models -------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # store hashed password

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), db.ForeignKey('user.username'), nullable=False)
    patient_name = db.Column(db.String(100))
    patient_age = db.Column(db.String(10))
    patient_gender = db.Column(db.String(10))
    patient_id = db.Column(db.String(50))
    doctor_name = db.Column(db.String(100))
    symptoms = db.Column(db.Text)
    original_image = db.Column(db.String(200))
    yolo_result = db.Column(db.Text)
    rf_result = db.Column(db.Text)

# ------------------- Roboflow Models -------------------
rf = Roboflow(api_key="kqlgTsdyBapHPYnoxznG")
model1 = rf.workspace().project("ecg-classification-ygs4v").version(1).model
model2 = rf.workspace().project("ecg_detection").version(3).model


# ------------------- Routes -------------------
@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):
        session['user'] = username
        return redirect(url_for('dashboard'))
    return render_template('login.html', error="Invalid Credentials")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match!")

        # Check if user already exists
        if User.query.filter((User.username == username) | (User.email == email)).first():
            return render_template('register.html', error="Username or Email already exists!")

        # Save user with hashed password
        hashed_pw = generate_password_hash(password)
        new_user = User(fullname=fullname, username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        session['user'] = username
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('home'))


@app.route('/upload')
def upload():
    if 'user' in session:
        return render_template('upload.html')
    return redirect(url_for('home'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (2213, 1572))
    cv2.imwrite(file_path, image)

    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_gender = request.form.get('patient_gender')
    patient_id = request.form.get('patient_id')
    doctor_name = request.form.get('doctor_name')
    patient_symptoms = request.form.get('patient_symptoms')

    ecg = ECG()
    user_image = ecg.getImage(file_path)
    gray_image = ecg.GrayImgae(user_image)
    gray_path = os.path.join(PROCESS_FOLDER, 'gray_image.png')
    cv2.imwrite(gray_path, gray_image * 255)

    leads = ecg.DividingLeads(user_image)
    ecg.PreprocessingLeads(leads)
    ecg.SignalExtraction_Scaling(leads)

    ecg_1d_signal = ecg.CombineConvert1Dsignal()
    ecg_reduced = ecg.DimensionalReduciton(ecg_1d_signal)
    ecg_prediction = ecg.ModelLoad_predict(ecg_reduced)

    pd.DataFrame(ecg_1d_signal).to_csv(os.path.join(PROCESS_FOLDER, '1d_signal.csv'), index=False)
    pd.DataFrame(ecg_reduced).to_csv(os.path.join(PROCESS_FOLDER, 'reduced_signal.csv'), index=False)

    result1 = model1.predict(file_path, confidence=40, overlap=30).json()
    result2 = model2.predict(file_path, confidence=40, overlap=30).json()

    def process_result(result, filename):
        predictions = result.get("predictions", [])
        if not predictions:
            return filename, ["No abnormality detected"]

        xyxy, confidence, class_id, labels = [], [], [], []
        predicted_classes = []

        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            confidence.append(pred["confidence"])
            class_id.append(pred["class_id"])
            labels.append(pred["class"])
            predicted_classes.append(pred["class"])

        xyxy = np.array(xyxy, dtype=np.float32)
        confidence = np.array(confidence, dtype=np.float32)
        class_id = np.array(class_id, dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        img = cv2.imread(file_path)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = box_annotator.annotate(scene=img, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        output_path = os.path.join(PROCESS_FOLDER, filename)
        cv2.imwrite(output_path, annotated)
        return filename, predicted_classes

    pred_img1, predicted_classes1 = process_result(result1, 'prediction1.jpg')
    pred_img2, predicted_classes2 = process_result(result2, 'prediction2.jpg')

    ecg_outputs = {
        'gray': 'process/gray_image.png',
        'lead_12': 'process/Leads_1-12_figure.png',
        'lead_13': 'process/Long_Lead_13_figure.png',
        'preprocessed_12': 'process/Preprossed_Leads_1-12_figure.png',
        'preprocessed_13': 'process/Preprossed_Leads_13_figure.png',
        'contour': 'process/Contour_Leads_1-12_figure.png'
    }

    # Save prediction to DB
    new_prediction = Prediction(
        username=session['user'],
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        patient_id=patient_id,
        doctor_name=doctor_name,
        symptoms=patient_symptoms,
        original_image=file_path,
        yolo_result=str(predicted_classes1 + predicted_classes2),
        rf_result=str(ecg_prediction)
    )
    db.session.add(new_prediction)
    db.session.commit()

    return render_template(
        'result.html',
        original=file.filename,
        pred1=pred_img1,
        pred2=pred_img2,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        patient_id=patient_id,
        doctor_name=doctor_name,
        patient_symptoms=patient_symptoms,
        predicted_classes1=predicted_classes1,
        predicted_classes2=predicted_classes2,
        ecg_prediction=ecg_prediction,
        ecg_outputs=ecg_outputs,
        ecg_1d_signal=pd.DataFrame(ecg_1d_signal),
        ecg_reduced=pd.DataFrame(ecg_reduced)
    )


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


@app.route('/process/<path:filename>')
def serve_process_file(filename):
    return send_from_directory(PROCESS_FOLDER, filename)


@app.route('/uploads/<path:filename>')
def serve_uploads_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ------------------- Run -------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # create tables if not exists
    app.run(debug=True)
