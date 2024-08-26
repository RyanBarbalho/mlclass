import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=15)

# Train the KNN model
knn.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)