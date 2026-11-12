"""Image conversion and PNG output."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Convert full-range YCbCr (0-255) to RGB uint8."""
    red = y + 1.402 * (cr - 128.0)
    green = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    blue = y + 1.772 * (cb - 128.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _text_chunk(keyword: str, text: str) -> bytes:
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    return _png_chunk(b"tEXt", payload)


def write_png(
    path: Path,
    rgb: np.ndarray,
    scale: int = 1,
    title: str | None = None,
) -> None:
    """Write an RGB uint8 array as a PNG, optionally nearest-neighbour scaled.

    When ``title`` is given it is stored as a PNG ``tEXt`` chunk so the output
    carries a human-readable label.
    """
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + (_text_chunk("Title", title) if title else b"")
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)
