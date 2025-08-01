import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

class ASLDataset(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X, float); self.y=torch.tensor(y, long)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

class Model(nn.Module):
    def __init__(self, inp, cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,cls)
        )
    def forward(self,x): return self.net(x)

df = pd.read_csv("asl_landmarks.csv")
X = df.drop("label", axis=1)
le = LabelEncoder()
y = le.fit_transform(df["label"])
joblib.dump(le, "label_encoder.joblib")
classes = len(set(y))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ds_tr = ASLDataset(X_train,y_train); ds_te=ASLDataset(X_test,y_test)
dl_tr, dl_te = DataLoader(ds_tr,32,True), DataLoader(ds_te,32)

m = Model(X_train.shape[1], classes)
opt = torch.optim.Adam(m.parameters(),0.001); lossfn = nn.CrossEntropyLoss()

for ep in range(20):
    m.train()
    total=0
    for xb,yb in dl_tr:
        opt.zero_grad()
        loss=lossfn(m(xb),yb); loss.backward(); opt.step()
        total+=loss.item()
    print(f"Epoch {ep+1} loss={total:.2f}")

# Eval
m.eval(); v=0; tot=0
with torch.no_grad():
    for xb,yb in dl_te:
        preds=m(xb).argmax(1)
        v+=(preds==yb).sum().item()
        tot+=len(yb)
print(f"Acc: {v/tot*100:.2f}%")

torch.save(m.state_dict(), "asl_model.pt")
print("✅ Model saved.")
