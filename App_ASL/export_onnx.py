import torch, onnx
from train_model import Model
model = Model(63,26)
model.load_state_dict(torch.load("asl_model.pt", map_location="cpu"))
model.eval()
dummy = torch.randn(1,63)
torch.onnx.export(model, dummy, "asl_model.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)
print("✅ Exported ASL model to ONNX.")
