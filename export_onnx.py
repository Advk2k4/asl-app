import torch
import torch.nn as nn
import numpy as np
import onnx

# Define the same model as in train_model.py
class ASLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26)  # 26 ASL letters A–Y excluding J and Z
        )

    def forward(self, x):
        return self.model(x)

# Load trained model weights
model = ASLClassifier()
model.load_state_dict(torch.load("asl_model.pt"))
model.eval()

# Dummy input with correct shape
dummy_input = torch.randn(1, 63)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "asl_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

# Verify ONNX export
import onnxruntime as ort
session = ort.InferenceSession("asl_model.onnx")
output = session.run(None, {"input": np.random.randn(1, 63).astype(np.float32)})
print("✅ Exported model to asl_model.onnx")
print("ONNX model verification successful. Output shape:", output[0].shape)
