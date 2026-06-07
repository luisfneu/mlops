#!/usr/bin/env python
"""Gera os splits de dados sem subir a stack — útil em CI ou primeiro setup."""
from __future__ import annotations

from src.data.load import load_raw
from src.data.preprocess import split_and_save


def main() -> None:
    df = load_raw()
    print(f"Dataset shape: {df.shape}")
    paths = split_and_save(df)
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
