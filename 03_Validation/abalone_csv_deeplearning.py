import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#carregar a base de dados
abalone_df = pd.read_csv("abalone_dataset.csv")

#encode para a coluna type
label_encoder = LabelEncoder()

abalone_df['type'] = label_encoder.fit_transform(abalone_df['type'])
abalone_df["sex"] = label_encoder.fit_transform(abalone_df['sex'])

#separar os dados entre features e target
X = abalone_df.drop('type', axis=1)
y = abalone_df['type']

#dividir os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#normalizar os dados
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#converter a variavel target para categorica
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#criar o modelo
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax')) # 3 classes para type 1,2,3

#compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#treinar o modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.4)

#avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.close()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_loss.png')
plt.close()