"""
Train and export the Fashion-MNIST CNN model.
This script trains the final CNN architecture and saves it to model/fashion_mnist_cnn.h5
"""

import os
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from sklearn.model_selection import KFold
from keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np

N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
BASE_LOG_DIR = "logs/fashion_mnist_kfold"
BASE_MODEL_DIR = "models/on_the_go/"

def load_format_dataset():
    """Load and format the Fashion-MNIST dataset."""
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_X, train_y, test_X, test_y


def preprocess_images(train_X, test_X):
    """Normalize images to 0-1 range."""
    train_nom = train_X.astype('float32')
    test_nom = test_X.astype('float32')
    train_nom = train_nom * 1. / 255.
    test_nom = test_nom * 1. / 255.
    return train_nom, test_nom

def specify_model():
    """Define the CNN architecture."""
    model = models.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))
    # 1st layer: Conv2D + MaxPooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                             kernel_initializer='he_uniform', 
                             padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    # 2nd layer: Conv2D + MaxPooling + Dropout
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # 3rd layer: Conv2D + MaxPooling + Dropout
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def cross_validation(trainX, trainY, BASE_LOG_DIR):
    """Train and save the model."""
    # K-Fold Setup
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    scores = []
    histories = []
    
    # Unique run ID for TensorBoard root
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation...\n")
    
    for train_idx, val_idx in kfold.split(trainX):
        # 1. Split Data for this fold
        X_train, X_val = trainX[train_idx], trainX[val_idx]
        y_train, y_val = trainY[train_idx], trainY[val_idx]
        
        # 2. Create Unique Paths for this fold
        fold_log_dir = os.path.join(BASE_LOG_DIR, f"run_{run_id}_fold_{fold_no}")
        fold_model_path = os.path.join(BASE_MODEL_DIR, f"fashion_mnist_cnn_fold_{fold_no}.keras")

        # 3. Define model
        print(f"Building model... {fold_no}/{N_FOLDS}")
        model = specify_model()
    
        # Callbacks
        checkpoint = ModelCheckpoint(
            fold_model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            mode='max', 
            restore_best_weights=True
        )
        tensorboard = TensorBoard(
            log_dir=fold_log_dir,
            histogram_freq=0,
            write_graph=True
        )
    
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[tensorboard, checkpoint, early_stopping],
            verbose=1
        )
        
        # 6. Evaluate
        _, acc = model.evaluate(X_val, y_val, verbose=2)
        print(f"> Fold {fold_no} Accuracy: {acc * 100:.2f}%")
        scores.append(acc)
        histories.append(history)
        
        # 7. Cleanup Memory
        tf.keras.backend.clear_session()
        fold_no += 1

    # --- Final Summary ---
    print("\n" + "="*30)
    print("CROSS-VALIDATION COMPLETE")
    print("="*30)
    print(f"Mean Accuracy:  {np.mean(scores)*100:.2f}% (+- {np.std(scores)*100:.2f}%)")
    print(f"Models saved to: {BASE_MODEL_DIR}/")
    print(f"Logs saved to:   {BASE_LOG_DIR}/")
    print("\nTo view TensorBoard:")
    print(f"%tensorboard --logdir {BASE_LOG_DIR}")

    return scores, histories


def train_final_model(trainX, trainY, log_dir, model_path):
    """
    Trains the final model on the FULL dataset for API deployment.
    """
    print("\n" + "="*40)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*40)
    
    # 1. Instantiate fresh model with same architecture
    final_model = specify_model()
    final_model.summary()
    
    # 2. Callbacks (No validation split needed for final training usually, 
    #    but we keep EarlyStopping logic based on training loss or a small holdout if desired)
    #    For maximum data usage, we train on ALL data. 
    #    We rely on the CV score for performance guarantee.
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='accuracy', # Monitor training accuracy since no val set
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False # Save full model
    )
    
    # Optional: EarlyStopping on training loss to prevent waste, 
    # but be careful not to stop too early without validation data.
    # Often omitted when training on full data if epochs are conservative.
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=12,
        mode='min',
        restore_best_weights=True
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # 3. Fit on FULL data (No validation_split)
    # We use the original trainX/trainY which contains 100% of available data
    history = final_model.fit(
        trainX, trainY,
        epochs=20, # Slightly higher epochs often needed as no validation stop
        batch_size=32,
        callbacks=[checkpoint, early_stopping, tensorboard],
        verbose=1
    )
    
    print(f"\n✅ Final model saved to: {model_path}")
    return final_model, history
    

if __name__ == '__main__':

    # Create model directory
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    # Create logs directory
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    
    # Load and preprocess data
    print("Loading dataset...")
    trainX, trainY, testX, testY = load_format_dataset()
    trainX, testX = preprocess_images(trainX, testX)
    
    # 1. Run Cross-Validation (To verify architecture)
    scores, histories = cross_validation(trainX, trainY, BASE_LOG_DIR)
    print(f"CV Mean Accuracy: {np.mean(scores)*100:.2f}%")
    
    # 2. Train Final Model on 100% of data for API
    final_log_dir = os.path.join(BASE_LOG_DIR, "final_model_run")
    final_model_path = os.path.join(BASE_MODEL_DIR, "fashion_mnist_api_final.keras")
    
    final_model, _ = train_final_model(trainX, trainY, final_log_dir, final_model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = final_model.evaluate(testX, testY, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')
