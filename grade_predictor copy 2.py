import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('StudentsPerformance_with_headers.csv')

# # Features to duplicate
# features_to_duplicate = ['Additional work', 'Weekly study hours', 'Reading frequency', 
#                          'Attendance to the seminars/conferences related to the department', 'Attendance to classes', 
#                          'Preparation to midterm exams 1', 'Preparation to midterm exams 2', 
#                          'Taking notes in classes', 'Listening in classes']

# # Check and duplicate the selected features
# for feature in features_to_duplicate:
#     if feature in data.columns:
#         for i in range(3):  # Duplicate each feature 3 times, for example
#             data[f'{feature}_dup_{i}'] = data[feature]
#     else:
#         print(f"Feature '{feature}' not found in the DataFrame.")

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

# # Output predictions and input sets
# for i in range(len(X_test)):
#     input_set = X_test.iloc[i]  # Get the input set
#     prediction = preds[i]      # Get the corresponding prediction
#     print(f"Input Set {i + 1}: {input_set}")
#     print(f"Predicted Output {i + 1}: {prediction}\n")

# Example input data (replace with your actual data and feature names)
input_data = {
    'STUDENT ID': ["Student111"],
    'Student Age': [2],
    'Sex': [2],
    'Graduated high-school type': [3],
    'Scholarship type': [3],
    'Additional work': [1],
    'Regular artistic or sports activity': [2],
    'Do you have a partner': [2],
    'Total salary if available': [1],
    'Transportation to the university': [1],
    'Accommodation type in Cyprus': [1],
    'Mother’s education': [1],
    'Father’s education ': [2],
    'Number of sisters/brothers': [3],
    'Parental status': [1],
    'Mother’s occupation': [2],
    'Father’s occupation': [5],
    'Weekly study hours': [3],
    'Reading frequency.1': [2],
    'Reading frequency': [2],
    'Attendance to the seminars/conferences related to the department': [1],
    'Impact of your projects/activities on your success': [1],
    'Attendance to classes': [1],
    'Preparation to midterm exams 1': [1],
    'Preparation to midterm exams 2': [1],
    'Taking notes in classes': [3],
    'Listening in classes': [2],
    'Discussion improves my interest and success in the course': [1],
    'Flip-classroom': [2],
    'Cumulative grade point average in the last semester (/4.00)': [1],
    'Expected Cumulative grade point average in the graduation (/4.00)': [1],
    'COURSE ID': [1]
}

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