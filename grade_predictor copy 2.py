import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask import Flask, request, render_template

# Load data
data = pd.read_csv('StudentsPerformance_with_headers.csv')

# Preprocess data
categorical_features = data.select_dtypes(include=['object']).columns
numerical_features = data.select_dtypes(exclude=['object']).columns.drop('GRADE')

# Define preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model
model = RandomForestRegressor(n_estimators=101, random_state=9060)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Separate target from predictors
y = data['GRADE']
X = data.drop('GRADE', axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model 
pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = pipeline.predict(X_test)

# Evaluate the model
score = mean_squared_error(y_test, preds)
print('MSE:', score)

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')  # Your HTML file

@app.route('/predict_grade', methods=['POST'])
def predict_grade():
    # Extracting form data
    input_data = {
    'STUDENT ID': [request.form['student_id']],
    'Student Age': [request.form['student_age']],
    'Sex': [request.form['student_sex']],
    'Graduated high-school type': [request.form['highschool_type']],
    'Scholarship type': [request.form['scholarship_type']],
    'Additional work': [request.form['student_work']],
    'Regular artistic or sports activity': [request.form['student_activity']],
    'Do you have a partner': [request.form['partner']],
    'Total salary if available': [request.form['salary']],
    'Transportation to the university': [request.form['transportation']],
    'Accommodation type in Cyprus': [request.form['accomodation']],
    'Mother’s education': [request.form['mother_education']],
    'Father’s education ': [request.form['father_education']],
    'Number of sisters/brothers': [request.form['siblings']],
    'Parental status': [request.form['parental_status']],
    'Mother’s occupation': [request.form['mother_occupation']],
    'Father’s occupation': [request.form['father_occupation']],
    'Weekly study hours': [request.form['study_hours']],
    'Reading frequency.1': [request.form['reading_frequency_non_sci']],
    'Reading frequency': [request.form['reading_frequency_sci']],
    'Attendance to the seminars/conferences related to the department': [request.form['attendance_seminars']],
    'Impact of your projects/activities on your success': [request.form['project_impact']],
    'Attendance to classes': [request.form['class_attendance']],
    'Preparation to midterm exams 1': [request.form['prep_midterm1']],
    'Preparation to midterm exams 2': [request.form['prep_midterm2']],
    'Taking notes in classes': [request.form['taking_notes']],
    'Listening in classes': [request.form['listening_classes']],
    'Discussion improves my interest and success in the course': [request.form['discussion_impact']],
    'Flip-classroom': [request.form['flip_classroom']],
    'Cumulative grade point average in the last semester (/4.00)': [request.form['last_sem_gpa']],
    'Expected Cumulative grade point average in the graduation (/4.00)': [request.form['expected_gpa']],
    'COURSE ID': [request.form['course_id']]
    }

    # Here, you can now use `input_data` with your prediction model
    # For example:
    # predicted_grade = your_model.predict(input_data)
    
    # Return or render a template with the prediction result
    # return f'Predicted Grade: {predicted_grade}'

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Use the trained pipeline to make a prediction
    predicted_grade = pipeline.predict(input_df)

    # Print the prediction
    print('Predicted GRADE:', predicted_grade[0])

    # Output predictions and input sets
    input_set = input_df.iloc[0]  # Get the input set
    prediction = predicted_grade[0]      # Get the corresponding prediction
    print(f"Input Set {0}: {input_set}")
    print(f"Predicted Output {0}: {prediction}\n")

    pred = round(prediction)

    output = get_grade(pred)

    return output 

def get_grade(grade_number):
    grade_switch = {
        0: 'Fail',
        1: 'DD',
        2: 'DC',
        3: 'CC',
        4: 'CB',
        5: 'BB',
        6: 'BA',
        7: 'AA'
    }

    return grade_switch.get(grade_number, "Invalid Grade")

if __name__ == "__main__":
    app.run(debug=True)