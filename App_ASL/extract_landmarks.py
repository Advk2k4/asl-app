import os, cv2, pandas as pd
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
rows = []

for label in os.listdir("data/images"):
    if not label.isalpha(): continue
    for imgfile in os.listdir(f"data/images/{label}")[:200]:
        img = cv2.imread(f"data/images/{label}/{imgfile}")
        if img is None: continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            row = [label] + [coord for p in lm.landmark for coord in (p.x, p.y, p.z)]
            rows.append(row)

cols = ["label"] + [f"{i}_{axis}" for i in range(21) for axis in ("x","y","z")]
pd.DataFrame(rows, columns=cols).to_csv("asl_landmarks.csv", index=False)
print("✅ Landmarks saved.")
