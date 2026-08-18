import os
import shutil
import torch
import subprocess
import onnx
import onnxsim

print("Step 1. Loading MiDaS v2.1 Small")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
onnx_raw = "midas_small.onnx"
onnx_sim = "midas_small_sim.onnx"

print("Step 2. Export into ONNX")
torch.onnx.export(
    model,
    dummy_input,
    onnx_raw,
    input_names=["input"],
    output_names=["depth"],
    opset_version=11
)

print("[Step 3. Graph simplification (onnxsim)")
model_proto = onnx.load(onnx_raw)
model_simplified, check = onnxsim.simplify(model_proto)
assert check, "Errror validation onnxsim"
onnx.save(model_simplified, onnx_sim)

print("Step 4. Convertation into NCNN with FP16")

subprocess.run(["pnnx", onnx_sim, "inputshape=[1,3,256,256]"], check=True)
ncnnoptimize_bin = shutil.which("ncnnoptimize") or os.path.expanduser(
    "~/ar_android_setup/ncnn/build/install/bin/ncnnoptimize"
)
subprocess.run([
    ncnnoptimize_bin,
    "midas_small_sim.ncnn.param",
    "midas_small_sim.ncnn.bin",
    "midas_fp16.param",
    "midas_fp16.bin",
    "65536"
], check=True)

print("\nDone: 4 steps from 4")

