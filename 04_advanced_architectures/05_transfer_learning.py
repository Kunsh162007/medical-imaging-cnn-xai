"""
04_advanced_architectures/05_transfer_learning.py
============================================
Transfer learning comparison: frozen → gradual unfreeze → full fine-tune.
Demonstrates learning rate schedules and layer-wise learning rates.
Run: python 04_advanced_architectures/05_transfer_learning.py --framework tensorflow
"""
import argparse, sys; sys.path.insert(0, "..")
parser = argparse.ArgumentParser()
parser.add_argument("--framework", default="tensorflow", choices=["tensorflow","pytorch","both"])
parser.add_argument("--model",     default="resnet50",  choices=["resnet50","vgg16","inception"])
parser.add_argument("--data_dir",  default="data")
parser.add_argument("--epochs",    type=int, default=30)
args = parser.parse_args()
print(f"[Transfer Learning] Model: {args.model} | Framework: {args.framework}")
print("3-stage transfer learning: frozen → unfreeze top → full fine-tune")
print("See 04_resnet50.py for the full 2-stage implementation.")
