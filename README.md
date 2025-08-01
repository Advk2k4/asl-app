# ASL Snapshot Recognition App

A computer vision application to recognize American Sign Language (ASL) letters using static hand gesture snapshots. Users can capture a webcam photo of a hand sign, and the app predicts the corresponding ASL letter using a trained ONNX neural network model.

## 📦 Project Structure

- `data_preprocessing.py` – Extracts and normalizes landmarks from dataset
- `train_model.py` – Trains a PyTorch MLP model
- `export_onnx.py` – Converts model to ONNX
- `app.py` – Streamlit UI with webcam snapshot support

## 🚀 Usage

1. Download ASL dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Run:
```
python data_preprocessing.py
python train_model.py
python export_onnx.py
streamlit run app.py
```

## ⚠️ Limitations

- Only supports static gestures (A–Z excluding J, Z)
- One hand per frame, clearly visible