import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

#class labels for CIFAR-10
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]


def load_and_preprocess_data():
    """Load CIFAR-10, normalize pixels, and one-hot encode target labels."""
    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #flatten y vectors
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    #scale pixel values to [0.0, 1.0]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    #one-hot encode targets for categorical crossentropy
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return X_train, y_train, y_train_cat, X_test, y_test, y_test_cat


def build_neural_network():
    """Build a feedforward neural network architecture with Keras"""
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def plot_training_history(history):
    """Plot Loss and Accuracy curves for Train vs. Validation across epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    #loss plot
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Test Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    #accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Test Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=True)


def evaluate_learning_rates(X_train, y_train_cat, X_test, y_test_cat, learning_rates=[0.001, 0.01, 0.1]):
    """Experiment with different SGD learning rates over 10 epochs."""
    print("\n--- Running Learning Rate Experiments ---")
    
    for lr in learning_rates:
        print(f"\nTraining model with Learning Rate = {lr}")
        model = build_neural_network()
        model.compile(
            optimizer=SGD(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            X_train, y_train_cat,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            verbose=1
        )
        
        final_loss = history.history['val_loss'][-1]
        final_acc = history.history['val_accuracy'][-1]
        print(f"Result for lr={lr}: Final Test Loss = {final_loss:.4f}, Final Test Acc = {final_acc:.4f}")


def plot_confusion_matrix_and_report(model, X_test, y_test):
    """Generate and display classification report and confusion matrix"""
    print("\nEvaluating model on test dataset...")
    
    #predict class probabilities and take argmax for predicted labels
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    #print Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES))

    #plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred_classes,
        display_labels=CLASS_NAMES,
        ax=ax
    )
    plt.title("Neural Network Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=True)


def main():
    try:
        #load and preprocess data
        X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = load_and_preprocess_data()

        #primary model (Optimal LR = 0.01)
        model = build_neural_network()
        model.summary()

        model.compile(
            optimizer=SGD(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        #train model for 50 Epochs
        print("\nTraining primary model for 50 epochs...")
        history = model.fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            verbose=1
        )

        #display training curves
        plot_training_history(history)

        #Classification Report & Confusion Matrix
        plot_confusion_matrix_and_report(model, X_test, y_test)

        #Learning Rate Comparison Experiment
        evaluate_learning_rates(X_train, y_train_cat, X_test, y_test_cat)

    except Exception as e:
        print(f"An error occurred during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()