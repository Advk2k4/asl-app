import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
labels = []
landmark_list = []

DATASET_PATH = "asl_alphabet_train/asl_alphabet_train"

# Fix: Skip hidden/system files
LABELS = sorted([
    d for d in os.listdir(DATASET_PATH)
    if not d.startswith('.') and os.path.isdir(os.path.join(DATASET_PATH, d))
])

for label in tqdm(LABELS):
    dir_path = os.path.join(DATASET_PATH, label)
    count = 0
    for file in os.listdir(dir_path):
        if not file.endswith('.jpg') or count >= 200:
            continue
        file_path = os.path.join(dir_path, file)
        image = cv2.imread(file_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            flattened = []
            for lm in hand_landmarks.landmark:
                flattened.extend([lm.x, lm.y, lm.z])
            landmark_list.append(flattened)
            labels.append(label)
            count += 1

# Ensure at least some data was collected
if len(landmark_list) > 0:
    df = pd.DataFrame(landmark_list)
    df['label'] = labels
    df.to_csv("asl_landmarks.csv", index=False)
    print(f"✅ Extracted {len(landmark_list)} samples and saved to asl_landmarks.csv")
else:
    print("❌ No landmarks were extracted. Check your dataset.")
