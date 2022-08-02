import onnx
import tensorflow as tf
import torch
from onnx_tf.backend import prepare

from unet_old import UNet

"""
net = UNet(n_channels=1, n_classes=2)
net.eval()
net.load_state_dict(torch.load("./important_pth/INTERRUPTED.pth"))
dummy_input = torch.randn(32, 1, 256, 256)
torch.onnx.export(
    net, dummy_input, "./checkpoints/ONNX_epoch0.onnx", opset_version=11,
)
"""
model = onnx.load("./checkpoints/ONNX_epoch0.onnx")
model = prepare(model)
model.export_graph("./checkpoints/TF_epoch0.pb")

model = tf.saved_model.load("./checkpoints/TF_epoch0.pb")
# tf.keras.models.save_model(model, "./checkpoints/TF_epoch0.h5")
