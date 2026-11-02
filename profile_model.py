import argparse
import json
from collections import Counter

import torch
from fvcore.nn import FlopCountAnalysis

from models.Models import DPCD


def main():
    parser = argparse.ArgumentParser(description="Profile FHRB-Net parameters and operations.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = DPCD().eval().to(device)
    inputs = tuple(torch.randn(1, 3, args.height, args.width, device=device) for _ in range(2))

    flop_error = None
    unsupported = {}
    try:
        analysis = FlopCountAnalysis(model, inputs)
        flops = analysis.total()
        unsupported = dict(Counter(analysis.unsupported_ops()))
    except Exception as exc:
        flops = None
        flop_error = f"{type(exc).__name__}: {exc}"
    result = {
        "model": "FHRB-Net-DPCD",
        "input": "two RGB images, batch=1",
        "image_size": [args.height, args.width],
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "fvcore_flops_per_pair": flops,
        "fvcore_convention": "one multiply-add is counted as one FLOP",
        "unsupported_ops": unsupported,
        "fvcore_error": flop_error,
        "note": "FFT and other unsupported operators, if listed, are excluded from the FLOP total.",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
