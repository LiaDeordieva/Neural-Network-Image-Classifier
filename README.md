# CIFAR-10 Image Classification with Keras & Multilayer Perceptron

A Feedforward Neural Network (Multilayer Perceptron) implemented in Python using **TensorFlow/Keras** to classify $32 \times 32$ color images from the **CIFAR-10** dataset across 10 object classes.

---

## Project Overview

This project demonstrates an end-to-end computer vision workflow using deep learning basics:
1. **Data Preprocessing:** Rescaling raw $RGB$ pixel values from $[0, 255]$ to $[0.0, 1.0]$ and applying one-hot encoding for multiclass target targets.
2. **Model Architecture:** Flattening $32 \times 32 \times 3$ image tensors into a 1D vector followed by a dense hidden layer with ReLU activation and a Softmax output layer.
3. **Training & Metrics:** Fitting over 50 epochs using Stochastic Gradient Descent (SGD) and tracking train vs. test loss/accuracy curves.
4. **Learning Rate Sensitivity Experiments:** Testing model convergence speed and stability across multiple learning rates ($\eta \in \{0.001, 0.01, 0.1\}$).
5. **Detailed Model Evaluation:** Computing class-wise precision, recall, F1-scores, and visualizing confusion matrices via `scikit-learn`.

---

## Target Classes

The model classifies images across the following 10 categories from CIFAR-10:

| Index | Label | Index | Label |
| :---: | :--- | :---: | :--- |
| **0** | Airplane | **5** | Dog |
| **1** | Automobile | **6** | Frog |
| **2** | Bird | **7** | Horse |
| **3** | Cat | **8** | Ship |
| **4** | Deer | **9** | Truck |

---

## Tech Stack

- **Language:** Python 3
- **Deep Learning Framework:** `TensorFlow` / `Keras`
- **Machine Learning & Metrics:** `scikit-learn`
- **Data Manipulation & Visualization:** `numpy`, `matplotlib`, `seaborn

**Install required dependencies:**
pip install tensorflow scikit-learn matplotlib seaborn numpy

---

## Result

<img width="1199" height="401" alt="Снимок экрана — 2026-08-22 в 17 18 03" src="https://github.com/user-attachments/assets/4836b466-1274-4193-ba2f-12b835e9c7d0" />

<img width="404" height="257" alt="Снимок экрана — 2026-08-22 в 17 19 25" src="https://github.com/user-attachments/assets/da3dbc8b-b986-48d4-aba5-c64dc2373544" />

<img width="796" height="759" alt="Снимок экрана — 2026-08-22 в 17 18 33" src="https://github.com/user-attachments/assets/e1b3d34b-cb01-44c5-930e-e401c456804f" />

---

## Network Architecture

```text
Input Image (32x32x3 RGB)
           │
           ▼
  [ Flatten Layer ]          --> Reshapes tensor to (3072,)
           │
           ▼
  [ Dense (128 units) ]      --> ReLU Activation
           │
           ▼
  [ Dense (10 units) ]       --> Softmax Activation
           │
           ▼
Output Probability Distribution
