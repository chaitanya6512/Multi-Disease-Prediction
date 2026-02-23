from flask import Blueprint, redirect, request, jsonify, session, current_app, render_template, url_for, flash
import pickle
import os
import pandas as pd
from datetime import datetime
from .models import Prediction, db
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Define the Blueprint
predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/test', methods=['GET'])
def test_prediction_route():
    return jsonify({'message': 'Predictions Blueprint is active!'}), 200

# Load Models
models_paths = os.path.join(os.getcwd(), 'models')

# Heart Disease Model
model_path_heart = os.path.join(models_paths, 'resnet.h5')
classes_heart = ['F- Fusion of Ventricular and Normal Beats', 
'M- Morphological Change in ECG', 
'N - Normal ECG', 'Q- Ventricular Flutter or Fibrillation', 'S- Supraventricular Premature Beat', 
'V - Premature Ventricular Contractions (PVCs)']
recommendation_heart =['For individuals with fusion of ventricular and normal beats, it is crucial to maintain a heart-healthy lifestyle. A diet rich in fruits, vegetables, and whole grains can help support overall cardiovascular function. Avoiding excessive caffeine and alcohol is essential, as these can trigger irregular heartbeats. Engaging in moderate exercise is beneficial, but strenuous activities should only be performed under medical supervision to prevent unnecessary strain on the heart.',
'Those experiencing morphological changes in their ECG should undergo regular cardiac monitoring and periodic ECG tests to track any developments. Managing underlying conditions such as hypertension and diabetes plays a vital role in maintaining heart health. A low-sodium, low-fat diet can help prevent further complications, and incorporating physical activity into daily routines can improve overall cardiovascular function. However, any changes in lifestyle should be made in consultation with a healthcare provider.',
'For individuals with a normal ECG, maintaining heart health is a lifelong commitment. A balanced diet, regular physical activity, and avoiding harmful habits such as smoking and excessive alcohol consumption are key to preventing cardiovascular diseases. Managing stress through relaxation techniques like meditation and ensuring regular health check-ups will help in early detection of any potential issues, keeping the heart functioning optimally.',
'Ventricular flutter or fibrillation is a serious condition that requires immediate medical attention, as it can be life-threatening. Individuals diagnosed with this condition should avoid high-stress activities and follow strict medical guidance, including prescribed medications and lifestyle adjustments. Cardiac rehabilitation programs can be highly beneficial in managing symptoms and improving heart function. Close monitoring and adherence to treatment plans are essential to preventing severe complications.',
'For those experiencing supraventricular premature beats, lifestyle modifications can help manage the condition effectively. Reducing the intake of caffeine and alcohol can minimize the occurrence of irregular beats. Stress management techniques, such as yoga and meditation, play an important role in maintaining stable heart rhythms. Additionally, regularly checking blood pressure and cholesterol levels ensures early detection and intervention if any abnormalities arise.',
'Individuals with premature ventricular contractions should focus on staying hydrated and avoiding excessive stimulants like caffeine. Engaging in moderate aerobic exercise can help strengthen the heart, but it is important to avoid overexertion. Monitoring and managing underlying conditions, such as electrolyte imbalances or high blood pressure, can reduce the frequency of PVCs. Consulting a healthcare professional for proper diagnosis and treatment is essential in maintaining long-term heart health.']
heart_model = load_model(model_path_heart)

# Pneumonia Model
model_path_pneumonia = os.path.join(models_paths, 'resnet_pneumonia.h5')
classes_pneumonia = ['NORMAL', 'PNEUMONIA']
recommendation_pneumonia = ['To maintain healthy lungs, practice deep breathing exercises, avoid smoking and pollutants, and stay hydrated. Ensure proper ventilation in your living space and get regular health check-ups. Strengthen your immunity with a balanced diet and stay up to date with flu and pneumonia vaccines as a preventive measure.','If diagnosed with pneumonia, ensure adequate rest and hydration, take prescribed antibiotics or antivirals, and use a humidifier to ease breathing. Avoid cold exposure, practice good hygiene, and get vaccinated to prevent future infections. Seek immediate medical attention if symptoms worsen.']
pneumonia_model = load_model(model_path_pneumonia)

# Retinopathy Model
model_path_retinopathy = os.path.join(models_paths, 'cnn.h5')
classes_retinopathy = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
recommendation_retinopathy = ['In the early stages of diabetic retinopathy, maintaining strict blood sugar control is essential to prevent disease progression. Regular eye check-ups with an ophthalmologist can help detect any changes before they become severe. A healthy diet, rich in antioxidants and low in processed sugars, along with regular exercise, can support overall eye health. Managing blood pressure and cholesterol levels also plays a key role in preventing further damage to the retinal blood vessels.',
'At the moderate stage, retinal damage is more noticeable, and close medical supervision is necessary. Controlling diabetes through proper medication, diet, and lifestyle changes can slow down further deterioration. It is advisable to avoid smoking and limit alcohol consumption, as these can worsen the condition. Regular eye exams, including dilated retinal imaging, should be scheduled to monitor disease progression and determine if early treatment interventions, such as laser therapy, are required.',
'For individuals without diabetic retinopathy, maintaining good diabetes management is key to preventing future complications. A balanced diet, combined with regular physical activity, can help keep blood sugar levels in check. Regular eye screenings are recommended, even if no symptoms are present, to detect any early signs of retinal changes. Lifestyle habits such as avoiding excessive screen time and wearing UV-protected eyewear can also contribute to long-term eye health.',
'Proliferative diabetic retinopathy is an advanced stage where abnormal blood vessels grow on the retina, increasing the risk of severe vision loss or blindness. Immediate medical intervention is required, including possible laser treatments (photocoagulation) or vitrectomy surgery to prevent further complications. Blood sugar, blood pressure, and cholesterol should be strictly controlled to reduce retinal damage. Patients must avoid strenuous activities that could lead to retinal hemorrhage and adhere to a structured eye care plan under medical supervision.',
'In severe cases, the risk of vision impairment is significantly high, requiring urgent medical attention. Intravitreal injections, laser therapy, or surgery may be necessary to manage the condition and prevent blindness. It is crucial to maintain strict blood sugar control, as uncontrolled diabetes accelerates retinal deterioration. Frequent eye exams, lifestyle modifications, and medical treatments are essential for managing the condition effectively. Patients should also be educated about recognizing warning signs such as sudden vision loss, floaters, or blurred vision, and seek immediate care if symptoms worsen.']
retinopathy_model = load_model(model_path_retinopathy)

# Diabetes Model and Encoders
with open(os.path.join(models_paths, 'random_forest_model.pkl'), 'rb') as f:
    diabetes_model = pickle.load(f)
with open(os.path.join(models_paths, 'encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)
with open(os.path.join(models_paths, 'standard_scaler.pkl'), 'rb') as f:
    standard_scaler = pickle.load(f)

# Utility Functions

def preprocess_image(image_path, target_size=(124, 124)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    x = np.array(img)
    return np.expand_dims(x, axis=0)

def preprocess_diabetes_data(input_data):
    df = pd.DataFrame([input_data])
    df['NewBMI'] = df['BMI'].apply(lambda x: 'Normal' if 18.5 <= x <= 24.9 else ('Overweight' if x > 24.9 else 'Underweight'))
    df['NewInsulinScore'] = df['Insulin'].apply(lambda x: 'Normal' if 16 <= x <= 166 else 'Abnormal')
    df['NewGlucose'] = df['Glucose'].apply(lambda x: 'Normal' if 70 <= x <= 99 else ('High' if x > 99 else 'Low'))

    encoded_data = encoder.transform(df[['NewBMI', 'NewInsulinScore', 'NewGlucose']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['NewBMI', 'NewInsulinScore', 'NewGlucose']))
    df = pd.concat([df.drop(columns=['NewBMI', 'NewInsulinScore', 'NewGlucose']), encoded_df], axis=1)

    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']] = standard_scaler.transform(
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']]
    )
    return df

# Heart Disease Prediction Route
@predictions_bp.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'GET':
        return render_template('predict_heart.html')

    uploaded_file = request.files.get('heartImage')
    if not uploaded_file or uploaded_file.filename == '':
        flash('No file selected or uploaded.', 'danger')
        return render_template('predict_heart.html')

    temp_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(temp_path)
    
    try:
        img_data = preprocess_image(temp_path)
        prediction = heart_model.predict(img_data)
        predicted_label = classes_heart[np.argmax(prediction)]
        predicted_recommendation_heart = recommendation_heart[np.argmax(prediction)]
        prediction_probs = prediction

        new_prediction = Prediction(
            user_id=session.get('user_id'),
            prediction_type='heart',
            input_data=uploaded_file.filename,
            result=predicted_label,
            created_at=datetime.utcnow()
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('heart_result.html', image_filename=uploaded_file.filename,predicted_label=predicted_label,prediction_prob=prediction_probs.tolist(),predicted_recommendation_heart=predicted_recommendation_heart)
    finally:
        os.remove(temp_path)

# Pneumonia Prediction Route
@predictions_bp.route('/predict/pneumonia', methods=['GET', 'POST'])
def predict_pneumonia():
    if request.method == 'GET':
        return render_template('predict_pneumonia.html')

    uploaded_file = request.files.get('pneumoniaImage')
    if not uploaded_file or uploaded_file.filename == '':
        flash('No file selected or uploaded.', 'danger')
        return render_template('predict_pneumonia.html')

    temp_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(temp_path)

    try:
        img_data = preprocess_image(temp_path)
        prediction = pneumonia_model.predict(img_data)
        predicted_label = classes_pneumonia[np.argmax(prediction)]
        predicted_recommendation_pneumonia = recommendation_pneumonia[np.argmax(prediction)]
        prediction_probs = prediction

        new_prediction = Prediction(
            user_id=session.get('user_id'),
            prediction_type='pneumonia',
            input_data=uploaded_file.filename,
            result=predicted_label,
            created_at=datetime.utcnow()
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('pneumonia_result.html', image_filename=uploaded_file.filename,predicted_label=predicted_label,prediction_prob=prediction_probs.tolist(), predicted_recommendation_pneumonia=predicted_recommendation_pneumonia)
    finally:
        os.remove(temp_path)

# Retinopathy Prediction Route
@predictions_bp.route('/predict/retinopathy', methods=['GET', 'POST'])
def predict_retinopathy():
    if request.method == 'GET':
        return render_template('predict_retinopathy.html')

    uploaded_file = request.files.get('retinopathyImage')
    if not uploaded_file or uploaded_file.filename == '':
        flash('No file selected or uploaded.', 'danger')
        return render_template('predict_retinopathy.html')

    temp_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(temp_path)

    try:
        img_data = preprocess_image(temp_path)
        prediction = retinopathy_model.predict(img_data)
        predicted_label = classes_retinopathy[np.argmax(prediction)]
        predicted_recommendation_retinopathy = recommendation_retinopathy[np.argmax(prediction)]
        prediction_probs = prediction

        new_prediction = Prediction(
            user_id=session.get('user_id'),
            prediction_type='retinopathy',
            input_data=uploaded_file.filename,
            result=predicted_label,
            created_at=datetime.utcnow()
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('retinopathy_result.html', image_filename=uploaded_file.filename,predicted_label=predicted_label,prediction_prob=prediction_probs.tolist(),predicted_recommendation_retinopathy=predicted_recommendation_retinopathy)
    finally:
        os.remove(temp_path)

# Diabetes Prediction Route
@predictions_bp.route('/predict/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'GET':
        return render_template('predict_diabetes.html')

    input_data = {
        'Pregnancies': int(request.form['Pregnancies']),
        'Glucose': int(request.form['Glucose']),
        'BloodPressure': int(request.form['BloodPressure']),
        'SkinThickness': int(request.form['SkinThickness']),
        'Insulin': int(request.form['Insulin']),
        'BMI': float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age': int(request.form['Age'])
    }

    try:
        features = preprocess_diabetes_data(input_data)
        y_pred = diabetes_model.predict(features)
        predicted_label = 'Diabetic' if y_pred[0] == 1 else 'Non-Diabetic'
        predicted_recommendation_diabetic = 'For individuals diagnosed with diabetes, maintaining strict blood sugar control is essential to prevent complications. A balanced diet rich in fiber, lean proteins, and healthy fats while avoiding processed sugars and carbohydrates helps regulate glucose levels. Regular physical activity, such as walking or moderate exercise, improves insulin sensitivity and overall health. Medications should be taken as prescribed, and regular monitoring of blood sugar levels is crucial. Additionally, routine check-ups with a doctor, including eye, kidney, and foot exams, help prevent long-term complications. Managing stress and getting adequate sleep also contribute to better diabetes management.' if y_pred[0] == 1 else 'For individuals without diabetes, adopting a healthy lifestyle is key to preventing the onset of the condition. Consuming a well-balanced diet, maintaining an active routine, and keeping body weight within a healthy range significantly reduce the risk of developing diabetes. Regular health screenings and blood sugar tests are recommended, especially for those with a family history of diabetes or other risk factors. Avoiding excessive sugar intake, processed foods, and leading a stress-free life through relaxation techniques like yoga and meditation can help maintain long-term metabolic health.'

        new_prediction = Prediction(
            user_id=session.get('user_id'),
            prediction_type='diabetes',
            input_data=str(input_data),
            result=predicted_label,
            created_at=datetime.utcnow()
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('diabetes_result.html', prediction=predicted_label,predicted_recommendation_diabetic=predicted_recommendation_diabetic)
    except Exception as e:
        current_app.logger.error(f"Error during diabetes prediction: {str(e)}")
        flash('An error occurred. Please try again later.', 'danger')
        return render_template('predict_diabetes.html')
