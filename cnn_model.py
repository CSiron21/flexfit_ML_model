import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_keypoints_and_labels(keypoints_dir):
    X = []
    y = []
    files = [f for f in os.listdir(keypoints_dir) if f.endswith('_keypoints.csv')]
    print(f"Found {len(files)} keypoints files: {files}")
    for f in files:
        label = f.split('_')[0]  # Prefix before first underscore
        df = pd.read_csv(os.path.join(keypoints_dir, f))
        X.append(df.values)
        y += [label] * len(df)
    X = np.vstack(X)
    y = np.array(y)
    return X, y, sorted(set(y))

def build_dense_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    keypoints_dir = 'train_data'
    X, y, label_list = load_keypoints_and_labels(keypoints_dir)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.1, random_state=42, stratify=y_enc)
    model = build_dense_model(input_dim=X.shape[1], num_classes=len(label_list))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)
    model.save('keypoints_dense_model.h5')
    # Save label list for metadata
    with open('keypoints_labels.txt', 'w') as f:
        for label in label_list:
            f.write(f'{label}\n')
    return model, label_list

def convert_to_tflite(model_path, label_list, tflite_path='keypoints_dense_model.tflite'):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    # Attach labels as metadata (simple append for demo)
    with open(tflite_path, 'ab') as f:
        f.write(b'\nLABELS:\n')
        for label in label_list:
            f.write(f'{label}\n'.encode())
    print(f'TFLite model exported to {tflite_path} with labels.')

if __name__ == '__main__':
    model, label_list = train_model()
    convert_to_tflite('keypoints_dense_model.h5', label_list) 