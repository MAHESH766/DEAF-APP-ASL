import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("asl_data.csv")
X = df.iloc[:, :-1].values  # Landmark data
y = df["label"].values  # Labels

# Encode labels as numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save trained model
model.save("asl_model.h5")
print("Model trained and saved as asl_model.h5")
