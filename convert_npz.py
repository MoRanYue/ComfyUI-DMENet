from __future__ import annotations

import argparse
from pathlib import Path

from dmenet_pytorch import convert_npz_to_pytorch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original TensorLayer DMENet .npz to PyTorch .pt")
    parser.add_argument("npz", type=Path, help="Path to original DMENet_BDCS.npz")
    parser.add_argument("output", type=Path, help="Output .pt path")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected variable is missing")
    args = parser.parse_args()

    report = convert_npz_to_pytorch(args.npz, args.output, strict=args.strict)
    print(f"Saved: {args.output}")
    print(f"Loaded tensors: {report.loaded}")
    if report.missing:
        print(f"Missing variables: {len(report.missing)}")
        for name in report.missing[:50]:
            print(f"  {name}")
        if len(report.missing) > 50:
            print("  ...")


if __name__ == "__main__":
    main()
