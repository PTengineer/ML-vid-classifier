"""
Quick QA script to validate model/dataset/transform shapes and forward passes.
Run:
    python train_testqa.py --module resnet
    python train_testqa.py --module slowfast
"""
import argparse
import traceback
import torch
from config import cfg
from models.model import build_model


def print_first_conv(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            print("first Conv3d weight.shape =", tuple(m.weight.shape))
            break


def synthetic_forward(module):
    cfg.ptv_module = module
    model = build_model(num_classes=2).eval()
    print_first_conv(model)
    B = 2
    if module == "resnet":
        x = torch.randn(B, 3, cfg.input_frames, cfg.input_size, cfg.input_size)
        try:
            y = model(x)
            print("resnet forward OK, out.shape:", y.shape)
        except Exception as e:
            print("resnet forward FAILED:", e)
            traceback.print_exc()
    else:
        fast = torch.randn(B, 3, cfg.input_frames, cfg.input_size, cfg.input_size)
        alpha = getattr(cfg, "slowfast_alpha", 4)
        slow = fast[:, :, ::alpha, :, :]
        try:
            y = model([slow, fast])
            print("slowfast forward OK, out.shape:", y.shape)
        except Exception as e:
            print("slowfast forward FAILED:", e)
            traceback.print_exc()


def sample_dataset_inspect():
    try:
        from datasets.video_dataset import RBCsDataset
        from transforms.video_transforms import test_transform
        ds = RBCsDataset(csv_file=cfg.train_csv, video_dir=cfg.data_root, transform=test_transform())
        v, lbl = ds[0]
        print("dataset sample type:", type(v))
        if isinstance(v, list):
            print("pathway shapes:", [getattr(x, 'shape', None) for x in v])
        else:
            print("sample shape:", getattr(v, 'shape', None), "dtype:", getattr(v, 'dtype', None))
    except Exception as e:
        print("Dataset inspect failed:", e)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["resnet", "slowfast"], default="resnet")
    args = parser.parse_args()
    print("Testing module:", args.module)
    synthetic_forward(args.module)
    print("\nInspecting dataset sample (if available):")
    sample_dataset_inspect()
