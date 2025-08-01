import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_DIR = "ASL_Alphabet_Dataset"
OUTPUT_CSV = "asl_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val
    return landmarks.flatten()

data = []
labels = []
valid_labels = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in ['J', 'Z']]

for label in tqdm(valid_labels, desc="Processing"):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.exists(label_dir):
        print(f"Missing folder: {label_dir}, skipping...")
        continue
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            annotated = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                results.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS
            )
            # st.image(annotated, caption="Detected Hand Landmarks", use_column_width=True)

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            try:
                norm_lm = normalize_landmarks(lm)
                data.append(norm_lm)
                labels.append(label)
            except:
                continue

hands.close()
df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} samples to {OUTPUT_CSV}")