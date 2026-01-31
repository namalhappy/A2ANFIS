# A2ANFIS: Attention-based ANFIS for Cyclone Strength Prediction

A2ANFIS is a hybrid deep learning model that combines **Convolutional Neural Networks (CNN)**, **Channel Attention mechanisms**, and an **Adaptive Neuro-Fuzzy Inference System (ANFIS)**. This architecture is designed to predict cyclone intensity (convective strength) by extracting spatial features from vapor cubes and processing them through a fuzzy logic-based decision layer.

## 🚀 Features

* **Hybrid Architecture**: Combines the feature extraction power of CNNs with the interpretability of Fuzzy Logic (ANFIS).
* **Channel Attention**: Implements SE-style attention to prioritize important feature channels in the input data.
* **GAN-based Training**: Utilizes a Discriminator with Spectral Normalization to improve the robustness of the Generator's predictions.
* **High-Value Weighting**: Includes a custom weighted L1 loss to improve accuracy on high-intensity events.
* **Performance Metrics**: Automatically calculates RMSE, MAE, R², and NSE (Nash-Sutcliffe Efficiency).

---

## 📂 Repository Structure

* `A2ANFIS.py`: The main training and inference script.
* `requirements.txt`: List of necessary Python dependencies.
* `data/`: Directory for your `train.h5` and `test.h5` files.
* `models/`: Output directory for trained model weights (`.pth`).
* `results/`: Directory for generated plots and metrics CSVs.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/namalhappy/A2ANFIS.git](https://github.com/namalhappy/A2ANFIS.git)
   cd A2ANFIS
