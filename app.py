import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
import cv2
import onnxruntime as ort
import joblib
import torch.nn.functional as F
import torch

# Load model and label encoder
session = ort.InferenceSession("asl_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
label_encoder = joblib.load("label_encoder.joblib")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Normalization function
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val
    return landmarks.flatten()

# Streamlit UI
st.title("Real-Time ASL Letter Recognition")
st.write("Show an ASL letter (A–Y, no J/Z) to the webcam")

class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            try:
                norm = normalize_landmarks(coords).astype(np.float32).reshape(1, -1)
                output = session.run([output_name], {input_name: norm})[0]
                probs = F.softmax(torch.tensor(output), dim=1).numpy()[0]
                top3 = probs.argsort()[-3:][::-1]
                top_labels = [(label_encoder.classes_[i], probs[i]) for i in top3]
                self.last_prediction = top_labels

                # Annotate image
                cv2.putText(img, f"1st: {top_labels[0][0]} ({top_labels[0][1]*100:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"2nd: {top_labels[1][0]} ({top_labels[1][1]*100:.1f}%)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(img, f"3rd: {top_labels[2][0]} ({top_labels[2][1]*100:.1f}%)", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            except Exception as e:
                cv2.putText(img, "Prediction Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

        return img

webrtc_streamer(key="asl", video_transformer_factory=ASLTransformer)
