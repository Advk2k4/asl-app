# 🧠 Real-Time ASL Sign Language Recognizer

A real-time, webcam-based American Sign Language (ASL) recognition system built with **Streamlit**, **OpenCV**, and **MediaPipe**. This app captures hand gestures and predicts corresponding ASL letters using a trained **KNN** model.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-%E2%9D%A4-red?logo=streamlit">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv">
</p>

---

## ✨ Features

- 🔴 **Live Webcam Input**  
  Detects hand gestures in real-time from your webcam.

- 🧠 **ASL Letter Prediction**  
  Uses hand landmarks to predict ASL alphabet letters with a trained KNN classifier.

- ⚙️ **Streamlit UI**  
  Lightweight, interactive UI running in-browser.

- 🧪 **Model Training Included**  
  Scripts to preprocess data and train your own KNN model.

---

## 🚀 Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/asl-sign-recognizer.git
cd asl-sign-recognizer
```

### 2. Install Dependencies

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧠 Train Your Own Model (Optional)

If you want to retrain the model:

```bash
cd training
python preprocess_asl.py   # Generates landmark CSV
python train_knn.py        # Trains and saves new model
```

Replace `asl_knn_model.joblib` with the newly trained one.

---

## 🌐 Deployment

You can deploy this app for free using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this project to a public GitHub repo  
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)  
3. Select your repo and `app.py` as the entrypoint  
4. Click "Deploy" 🎉

---

## 📸 Example

> *(Insert a screenshot or gif here)*  
> Showing a user doing hand gestures and the predicted letter appearing live.

---

## 📚 Tech Stack

- Python 3.9
- Streamlit
- OpenCV
- MediaPipe
- scikit-learn (KNN model)

---

## 🤝 Contributing

Pull requests are welcome. If you’d like to add more ASL signs or improve the model, feel free to fork the repo and open a PR.

---

## 📄 License

MIT License © Aadvik Mishra
