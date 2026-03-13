#!/usr/bin/env python3
"""
Render a colored Mandelbrot set image and save it as a binary PPM file.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from numba import njit, prange
except ImportError:
    np = None
    njit = None
    prange = None

PALETTE_SIZE = 4096


def _build_palette(size: int = PALETTE_SIZE) -> tuple[tuple[int, int, int], ...]:
    """
    Precompute the HSV -> RGB mapping once to avoid per-pixel costs.
    """

    palette = []
    for i in range(size):
        hue = (i / (size - 1) + 0.7) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        palette.append(
            (
                int(255 * r + 0.5),
                int(255 * g + 0.5),
                int(255 * b + 0.5),
            )
        )
    return tuple(palette)


PALETTE = _build_palette()

NUMBA_AVAILABLE = njit is not None and np is not None
if np is not None:
    PALETTE_ARRAY = np.array(PALETTE, dtype=np.uint8)
else:
    PALETTE_ARRAY = None

ENGINE_CHOICES = ("auto", "python", "numba")


def _is_in_cardioid_or_bulb(cr: float, ci: float) -> bool:
    """
    Fast analytic test for main cardioid and period-2 bulb membership.
    """

    # Check main cardioid
    cr_minus_quarter = cr - 0.25
    q = cr_minus_quarter * cr_minus_quarter + ci * ci
    if q * (q + cr_minus_quarter) <= 0.25 * ci * ci:
        return True

    # Check period-2 bulb
    cr_plus_one = cr + 1.0
    if cr_plus_one * cr_plus_one + ci * ci <= 0.0625:
        return True

    return False


def _render_python(
    width: int,
    height: int,
    max_iter: int,
    center: tuple[float, float],
    view_width: float,
    show_progress: bool = True,
) -> bytes:
    """
    Generate the raw RGB bytes for the Mandelbrot set image.
    """

    pixel_width = view_width / width
    view_height = pixel_width * height
    pixel_height = view_height / height

    min_real = center[0] - view_width / 2
    min_imag = center[1] - view_height / 2

    data = bytearray(width * height * 3)
    idx = 0

    palette = PALETTE
    palette_max = len(palette) - 1
    log = math.log
    log2 = log(2.0)
    bailout = 4.0
    float_max_iter = float(max_iter)

    if show_progress:
        progress_step = max(1, height // 100)
        next_progress = progress_step
    else:
        progress_step = 0
        next_progress = 0

    for y in range(height):
        ci = min_imag + y * pixel_height
        cr = min_real
        for _ in range(width):
            if _is_in_cardioid_or_bulb(cr, ci):
                data[idx] = 0
                data[idx + 1] = 0
                data[idx + 2] = 0
                idx += 3
                cr += pixel_width
                continue

            zr = 0.0
            zi = 0.0
            zr2 = 0.0
            zi2 = 0.0
            iteration = 0

            while iteration < max_iter and zr2 + zi2 <= bailout:
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                zr2 = zr * zr
                zi2 = zi * zi
                iteration += 1

            if iteration >= max_iter:
                data[idx] = 0
                data[idx + 1] = 0
                data[idx + 2] = 0
            else:
                modulus_sq = zr2 + zi2
                log_zn = 0.5 * log(modulus_sq)
                nu = log(log_zn / log2)
                smooth_iter = iteration + 1 - nu
                normalized = smooth_iter / float_max_iter
                if normalized < 0.0:
                    normalized = 0.0
                elif normalized > 1.0:
                    normalized = 1.0
                palette_index = int(normalized * palette_max + 0.5)
                r, g, b = palette[palette_index]
                data[idx] = r
                data[idx + 1] = g
                data[idx + 2] = b

            idx += 3
            cr += pixel_width

        if show_progress and (y + 1) >= next_progress:
            percent = (y + 1) * 100 / height
            sys.stdout.write(f"\rRendering: {percent:5.1f}%")
            sys.stdout.flush()
            next_progress = min(height, (y + 1) + progress_step)

    if show_progress:
        sys.stdout.write("\n")

    return bytes(data)


if NUMBA_AVAILABLE:

    @njit(fastmath=True, inline="always")
    def _is_in_cardioid_or_bulb_jit(cr: float, ci: float) -> bool:
        cr_minus_quarter = cr - 0.25
        q = cr_minus_quarter * cr_minus_quarter + ci * ci
        if q * (q + cr_minus_quarter) <= 0.25 * ci * ci:
            return True

        cr_plus_one = cr + 1.0
        if cr_plus_one * cr_plus_one + ci * ci <= 0.0625:
            return True

        return False

    @njit(parallel=True, fastmath=True)
    def _mandelbrot_kernel(
        data: np.ndarray,
        width: int,
        height: int,
        max_iter: int,
        min_real: float,
        min_imag: float,
        pixel_width: float,
        pixel_height: float,
        palette: np.ndarray,
        palette_max: int,
    ) -> None:
        bailout = 4.0
        log2 = math.log(2.0)
        float_max_iter = float(max_iter)

        for y in prange(height):
            ci = min_imag + y * pixel_height
            for x in range(width):
                cr = min_real + x * pixel_width

                if _is_in_cardioid_or_bulb_jit(cr, ci):
                    data[y, x, 0] = 0
                    data[y, x, 1] = 0
                    data[y, x, 2] = 0
                    continue

                zr = 0.0
                zi = 0.0
                zr2 = 0.0
                zi2 = 0.0
                iteration = 0

                while iteration < max_iter and zr2 + zi2 <= bailout:
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
                    zr2 = zr * zr
                    zi2 = zi * zi
                    iteration += 1

                if iteration >= max_iter:
                    data[y, x, 0] = 0
                    data[y, x, 1] = 0
                    data[y, x, 2] = 0
                else:
                    modulus_sq = zr2 + zi2
                    log_zn = 0.5 * math.log(modulus_sq)
                    nu = math.log(log_zn / log2)
                    smooth_iter = iteration + 1 - nu
                    normalized = smooth_iter / float_max_iter
                    if normalized < 0.0:
                        normalized = 0.0
                    elif normalized > 1.0:
                        normalized = 1.0
                    palette_index = int(normalized * palette_max + 0.5)
                    if palette_index < 0:
                        palette_index = 0
                    elif palette_index > palette_max:
                        palette_index = palette_max
                    color = palette[palette_index]
                    data[y, x, 0] = color[0]
                    data[y, x, 1] = color[1]
                    data[y, x, 2] = color[2]


def _render_numba(
    width: int,
    height: int,
    max_iter: int,
    center: tuple[float, float],
    view_width: float,
) -> bytes:
    if not NUMBA_AVAILABLE or PALETTE_ARRAY is None:
        raise RuntimeError(
            "Numba backend requested but numba or numpy is not installed."
        )

    pixel_width = view_width / width
    view_height = pixel_width * height
    pixel_height = view_height / height

    min_real = center[0] - view_width / 2
    min_imag = center[1] - view_height / 2

    data = np.empty((height, width, 3), dtype=np.uint8)
    _mandelbrot_kernel(
        data,
        width,
        height,
        max_iter,
        min_real,
        min_imag,
        pixel_width,
        pixel_height,
        PALETTE_ARRAY,
        PALETTE_ARRAY.shape[0] - 1,
    )

    return data.tobytes()


def render(
    width: int,
    height: int,
    max_iter: int,
    center: tuple[float, float],
    view_width: float,
    show_progress: bool = True,
    engine: str = "auto",
) -> bytes:
    """
    Dispatch to the fastest available renderer.
    """

    engine = engine.lower()
    if engine not in ENGINE_CHOICES:
        raise ValueError(f"Unsupported engine '{engine}'. Choose from {ENGINE_CHOICES}.")

    use_numba = False
    if engine == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "Numba backend requested but numba or numpy is not installed."
            )
        use_numba = True
    elif engine == "auto":
        use_numba = NUMBA_AVAILABLE

    if use_numba:
        if show_progress:
            print("Rendering with numba backend (progress reporting disabled).")
        return _render_numba(
            width=width,
            height=height,
            max_iter=max_iter,
            center=center,
            view_width=view_width,
        )

    return _render_python(
        width=width,
        height=height,
        max_iter=max_iter,
        center=center,
        view_width=view_width,
        show_progress=show_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a colored Mandelbrot image (writes binary PPM)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Image width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Image height in pixels (default: 720)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations per pixel (default: 500)",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=(-0.75, 0.0),
        metavar=("REAL", "IMAG"),
        help="Center of the view window (default: -0.75 0.0)",
    )
    parser.add_argument(
        "--view-width",
        type=float,
        default=3.5,
        help="Width of the complex plane portion to render (default: 3.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mandelbrot.ppm"),
        help="Output PPM file path (default: mandelbrot.ppm)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the row-by-row progress indicator",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--engine",
        choices=ENGINE_CHOICES,
        default="auto",
        help="Select rendering engine: auto, python, or numba (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb_data = render(
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
        center=tuple(args.center),
        view_width=args.view_width,
        show_progress=args.progress,
        engine=args.engine,
    )

    header = f"P6\n{args.width} {args.height}\n255\n".encode("ascii")
    args.output.write_bytes(header + rgb_data)
    print(f"Wrote Mandelbrot image to {args.output.resolve()}")


if __name__ == "__main__":
    main()
