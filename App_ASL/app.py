import streamlit as st, cv2, mediapipe as mp, numpy as np, onnxruntime
from PIL import Image

# Setup ONNX
sess = onnxruntime.InferenceSession("asl_model.onnx")
label_map = [chr(i) for i in range(65, 91)]  # A-Z

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=1)

def extract(img):
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks: return None
    lm = res.multi_hand_landmarks[0]
    arr = np.array([[p.x,p.y,p.z] for p in lm.landmark])
    arr -= arr[0]
    arr /= (np.linalg.norm(arr)+1e-6)
    return arr.flatten().astype(np.float32).reshape(1,-1)

st.title("📷 ASL Snapshot Recognition")
st.write("Click the button to capture and predict your sign")

cap = cv2.VideoCapture(0)
if st.button("Capture"):
    ret, frame = cap.read()
    if not ret: st.error("Camera not found"); cap.release(); st.stop()
    st.image(frame[:,:,::-1], caption="Captured", use_column_width=True)
    feats = extract(frame)
    if feats is None: st.warning("No hand detected"); cap.release(); st.stop()
    out = sess.run(None, {"input":feats})[0][0]
    idx = int(np.argmax(out))
    st.success(f"**{label_map[idx]}** (Confidence: {out[idx]*100:.1f}%)")
    cap.release()
