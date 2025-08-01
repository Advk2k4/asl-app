# ASL Snapshot Recognition App 🖐️📷

This project is a webcam-based American Sign Language (ASL) recognition system built with Python, MediaPipe, PyTorch, and Streamlit. It allows users to show a static hand gesture to the webcam and predicts the corresponding ASL letter using a trained machine learning model.

---

## 🎥 Demo Video

> _⬇️ Watch the full demo of the app in action here:_
- https://www.youtube.com/watch?v=_ZuMkJ-Eqhw&ab_channel=AadvikMishra

---

## 🧠 Features

- Live webcam snapshot input using Streamlit  
- Hand landmark extraction via MediaPipe  
- Landmark normalization for consistent model input  
- Trained ONNX model for inference  
- Softmax-based top-3 ASL letter predictions  
- Friendly UI with prediction confidence display

---

## 📁 Folder Structure

```
ASL-App/
├── app.py                    # Streamlit UI app
├── asl_model.onnx            # Trained ONNX model
├── label_encoder.joblib      # Class label encoder
├── data_preprocessing.py     # Extracts and normalizes landmarks from dataset
├── train_model.py            # Trains the PyTorch model
├── export_onnx.py            # Exports PyTorch model to ONNX
├── requirements.txt          # Dependencies
├── README.md                 # Project overview and instructions
```

---

## ⚙️ How to Run

1. Clone this repo:
```bash
git clone https://github.com/yourusername/ASL-App.git
cd ASL-App
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the app:
```bash
streamlit run app.py
```

5. A browser window will open at `http://localhost:8501`. Show an ASL hand sign, take a snapshot, and see predictions.

---

## 🔁 Retraining the Model (Optional)

If you'd like to retrain on your own data:

1. Prepare dataset under `ASL_Alphabet_Dataset/`
2. Run:
```bash
python data_preprocessing.py
python train_model.py
python export_onnx.py
```

This regenerates the ONNX model and label encoder based on updated data.

---

## 🧰 Built With

- **Python 3.11**
- **Streamlit** — frontend and webcam snapshot capture
- **MediaPipe** — hand landmark detection (21 keypoints)
- **PyTorch** — model training and export
- **ONNX Runtime** — inference during app runtime

---

## 🔤 ASL Classes Supported

This app currently supports 24 static ASL letters (A–Z excluding J and Z due to motion-based gestures).

---

## 👨‍💻 Authors

- **Aadvik Mishra**

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
