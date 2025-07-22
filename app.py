import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="ASL Sign Recognizer", layout="centered")
st.title("🧠 Real-Time American Sign Language Letter Recognition")
st.markdown("Show an American Sign Language letter (A, B, C...) with your hand and the app will recognize it!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ASLRecognizer(VideoProcessorBase):
    def __init__(self):
        self.model = joblib.load("asl_knn_model.joblib")
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.prediction = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            flattened = []
            for lm in hand_landmarks.landmark:
                flattened.extend([lm.x, lm.y, lm.z])

            if len(flattened) == 63:
                X = np.array(flattened).reshape(1, -1)
                self.prediction = model.predict(X)[0]
                cv2.putText(img, f"Letter: {self.prediction}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        return img

# Compatible RTC config for Streamlit Cloud
webrtc_streamer(
    key="asl",
    video_processor_factory=ASLRecognizer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    },
    async_processing=True
)
