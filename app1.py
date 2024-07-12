from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the diabetes dataset
df_diabetes = pd.read_csv('diabete.csv')

# Preprocess the diabetes dataset
X_diabetes = df_diabetes[['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                          'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness',
                          'Alopecia', 'Obesity']]
y_diabetes = df_diabetes['Outcome']

# Split the diabetes dataset into training and test sets
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.2, stratify=y_diabetes, random_state=2)

# Scale the features for the diabetes dataset
scaler_diabetes = StandardScaler()
X_train_scaled_diabetes = scaler_diabetes.fit_transform(X_train_diabetes)
X_test_scaled_diabetes = scaler_diabetes.transform(X_test_diabetes)

# Train the SVM model for diabetes prediction
svm_model_diabetes = SVC(kernel='linear', random_state=2)
svm_model_diabetes.fit(X_train_scaled_diabetes, y_train_diabetes)

y_pred_diabetes = svm_model_diabetes.predict(X_test_scaled_diabetes)
test_accuracy_diabetes = accuracy_score(y_test_diabetes, y_pred_diabetes)
print("Test Accuracy for Diabetes Prediction:", test_accuracy_diabetes * 100)

# Load the mental health dataset
df_mental_health = pd.read_csv('mental.csv')

# Separate features and target variable for mental health dataset
X_mental_health = df_mental_health.drop('Expert Diagnose', axis=1)
y_mental_health = df_mental_health['Expert Diagnose']

# Encode categorical variables for mental health dataset
label_encoders_mental_health = {}
for column in X_mental_health.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_mental_health[column] = le.fit_transform(X_mental_health[column])
    label_encoders_mental_health[column] = le
# print(label_encoders_mental_health)

# Handle missing values if any for mental health dataset
X_mental_health = X_mental_health.fillna(X_mental_health.mean())

# Standardize numerical features for mental health dataset
scaler_mental_health = StandardScaler()
X_scaled_mental_health = pd.DataFrame(scaler_mental_health.fit_transform(X_mental_health), columns=X_mental_health.columns)

# Split the mental health dataset into training and testing sets
X_train_mental_health, X_test_mental_health, y_train_mental_health, y_test_mental_health = train_test_split(X_scaled_mental_health, y_mental_health, test_size=0.2, random_state=42)

# Train the model for mental health prediction
model_mental_health = RandomForestClassifier()
model_mental_health.fit(X_train_mental_health, y_train_mental_health)

# Calculate accuracy on the test set for mental health prediction
y_pred_mental_health = model_mental_health.predict(X_test_mental_health)
test_accuracy_mental_health = accuracy_score(y_test_mental_health, y_pred_mental_health)
print("Test Accuracy for Mental Health Prediction:", test_accuracy_mental_health * 100)

# Load the heart disease dataset
data = pd.read_csv('heart.csv')
data['age'] = data['age'] // 365
X_heart = data.drop(columns=['cardio'])
y_heart = data['cardio']
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Initialize individual classifiers for heart disease prediction
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the individual classifiers
rf_classifier.fit(X_train_heart, y_train_heart)
# gb_classifier.fit(X_train_heart, y_train_heart)

# Initialize a voting classifier with the individual classifiers for heart disease prediction
# voting_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('gb', gb_classifier)], voting='soft')
# voting_classifier.fit(X_train_heart, y_train_heart)

# Make predictions on the test set for heart disease prediction
y_pred_heart = rf_classifier.predict(X_test_heart)
test_accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
print("Test Accuracy for Heart Disease Prediction:", test_accuracy_heart * 100)


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/mental_health')
def mental_health():
    return render_template('mental_health.html')
@app.route('/heart')
def mental_health():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.form['predict_type'] == 'diabetes':
        input_data = []

        # Extract Age
        input_data.append(int(request.form['age']))

        # Extract other symptoms
        input_data.extend([1 if request.form[symptom] == 'Yes' else 0 for symptom in
                           ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                            'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                            'muscle stiffness', 'Alopecia', 'Obesity']])

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data], columns=['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                                                       'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                                                       'Irritability', 'delayed healing', 'partial paresis',
                                                       'muscle stiffness', 'Alopecia', 'Obesity'])

        # Scale the input data
        input_X_scaled = scaler_diabetes.transform(input_df)

        # Predict using the SVM model
        prediction = svm_model_diabetes.predict(input_X_scaled)

        # Determine the prediction result
        if prediction[0] == 1:
            result = "Diabetic / ಮಧುಮೇಹ"
            advice = "You have been predicted as diabetic. It is important to monitor your blood sugar levels regularly, follow a healthy diet, engage in regular physical activity, and take medications as prescribed by your doctor."
            redirect_url = 'diabetic_advice.html'
        else:
            result = "Non-Diabetic"
            advice = "You have been predicted as non-diabetic. However, it is still important to maintain a healthy lifestyle by eating a balanced diet, exercising regularly, and avoiding unhealthy habits such as smoking and excessive drinking."
            redirect_url = 'non_diabetic_advice.html'

        # Redirect to the appropriate page based on the prediction result
        return redirect(url_for('advice', result=result, advice=advice, redirect_url=redirect_url))

    elif request.form['predict_type'] == 'mental_health':
        input_data = {column: request.form[column] for column in X_mental_health.columns}

        # Map form data to numerical values directly
        for column, le in label_encoders_mental_health.items():
            input_data[column] = le.transform([input_data[column]])[0]

        # Convert input data to a NumPy array
        input_data = np.array(list(input_data.values())).reshape(1, -1)

        # Standardize numerical features
        input_data = scaler_mental_health.transform(input_data)

        # Make prediction
        prediction = model_mental_health.predict(input_data)[0]

        return render_template('mental_res.html', prediction=prediction, accuracy=test_accuracy_mental_health)

    elif request.form['predict_type'] == 'heart_disease':
        age_days = int(request.form['age'])
        age_years = age_days // 365  # Convert age from days to years
        gender = int(request.form['gender'])
        height = float(request.form['height']) * 30.48
        weight = float(request.form['weight'])
        ap_hi = float(request.form['ap_hi'])
        ap_lo = float(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = 1 if 'smoke' in request.form else 0
        alco = 1 if 'alco' in request.form else 0
        active = 1 if 'active' in request.form else 0

        prediction = voting_classifier.predict([[age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
                                                  smoke, alco, active]])

        if prediction[0] == 0:
            result = "No Cardiovascular Disease"
        else:
            result = "Cardiovascular Disease"

        return render_template('heartresult.html', result=result, accuracy=test_accuracy_heart)

   


@app.route('/advice')
def advice():
    result = request.args.get('result')
    advice = request.args.get('advice')
    redirect_url = request.args.get('redirect_url')
    return render_template('advice.html', result=result, advice=advice, redirect_url=redirect_url)


if __name__ == '__main__':
    app.run(port=8080,debug=True)
