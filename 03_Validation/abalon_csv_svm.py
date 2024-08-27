import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Load the abalone dataset from the CSV file
abalone_df = pd.read_csv("abalone_dataset.csv")

# Encode the 'type' and 'sex' columns
label_encoder = LabelEncoder()
abalone_df['type'] = label_encoder.fit_transform(abalone_df['type'])
abalone_df['sex'] = label_encoder.fit_transform(abalone_df['sex'])

# Separate the data into features and target
X = abalone_df.drop('type', axis=1)
y = abalone_df['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm = svc = SVC(kernel = 'linear')

svm.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = svm.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)