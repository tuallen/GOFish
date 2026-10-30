# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Per-calibration curvature table used by complete admission.

Complete admission bounds the second derivative of every calibrated camera's
projection over 5-degree bands of the ray angle.
The bounds are computed once per calibration with outward-rounded interval
arithmetic and compiled into the renderer as
``include/3dgut/kernels/cuda/renderers/certifiedRemainderPackedKbTable.cuh``.

The renderer selects a table row by exact float32 equality of the active
camera parameters (focal length, principal point, distortion, maximum angle)
and raster size.  Rows are append-only: extending the table never changes the
rows of existing calibrations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Sequence

from . import qualified_interval as qualified


HEADER = struct.Struct("<8s8I3Q32s32s32s")
CAMERA = struct.Struct("<32s32s32s6Id")
BAND = struct.Struct("<ddd")
MAGIC = b"GFKBH02\0"
ENDIAN_SENTINEL = 0x01020304
FLAGS = 0x3F
HEADER_BYTES = 160
CAMERA_BYTES = 128
BAND_BYTES = 24

MODEL_IDS = {"OPENCV_FISHEYE": 1, "PINHOLE": 2}
MODEL_PARAMETER_COUNTS = {"OPENCV_FISHEYE": 8, "PINHOLE": 4}
SELECTOR_NATIVE_INTEGER_SCALABLE = 0
SELECTOR_ACTIVE_CORNER_UNIT_SCALE = 1

BAND_WIDTH = math.radians(5.0)
SMALL_AXIS_LIMIT = 0.01
SUBDIVISIONS = 128

HEADER_PATH = (Path(__file__).resolve().parents[1] / "include" / "3dgut" / "kernels"
               / "cuda" / "renderers" / "certifiedRemainderPackedKbTable.cuh")


class TableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeCamera:
    """Camera parameters exactly as the renderer receives them."""

    model: str
    width: int
    height: int
    parameters: tuple[float, ...]  # fx, fy, cx, cy[, k1..k4]
    max_angle: float               # radians; ignored for PINHOLE

    def key(self) -> tuple:
        words = tuple(f32_word(value) for value in self.parameters)
        angle = 0 if self.model == "PINHOLE" else f32_word(self.max_angle)
        return (self.model, self.width, self.height, words, angle)


# ----------------------------------------------------------------------------
# float32 helpers

def f32(value: float) -> float:
    rounded = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    if not math.isfinite(rounded):
        raise TableError(f"{value!r} is not a finite float32")
    return rounded


def f32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def float_literal(value: float) -> str:
    return f32(value).hex() + "f"


def pinhole_theta_max(width: int, height: int, parameters: Sequence[float]) -> float:
    fx, fy, cx, cy = parameters
    radius = max(math.hypot((u - cx) / fx, (v - cy) / fy)
                 for u in (0.0, float(width)) for v in (0.0, float(height)))
    return math.nextafter(math.atan(radius), math.inf)


def certificate_max_angle(max_angle: float, model: str) -> float:
    """Largest double that rounds to the same float32 maxAngle.

    Covers every source value sharing the selector's float32 bits.  Fisheye
    endpoints stay strictly below pi, where the KB4 bound remains finite.
    """
    word = f32_word(max_angle)
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    if not value > 0.0 or word >= 0x7F7FFFFF:
        raise TableError("maxAngle has no finite positive float32 bin")
    successor = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    endpoint = math.nextafter((float(value) + float(successor)) * 0.5, -math.inf)
    if model == "OPENCV_FISHEYE":
        endpoint = min(endpoint, math.nextafter(math.pi, -math.inf))
    if f32_word(endpoint) != word:
        raise TableError("canonical maxAngle endpoint changed selector bits")
    return endpoint


# ----------------------------------------------------------------------------
# curvature bands

def band_edges(theta_max: float) -> list[float]:
    result = [0.0, min(SMALL_AXIS_LIMIT, theta_max)]
    value = BAND_WIDTH
    while value < theta_max:
        if value > result[-1]:
            result.append(value)
        value += BAND_WIDTH
    if theta_max > result[-1]:
        result.append(theta_max)
    return result


def pinhole_hbar(theta_lo: float, theta_hi: float, fx: float, fy: float) -> float:
    interval = qualified.Interval(theta_lo, theta_hi)
    sin_i = qualified.sin_interval(interval)
    cos_i = qualified.cos_interval(interval)
    if not cos_i.lo > 0.0:
        raise TableError("pinhole band crosses the projection plane")
    sin2 = qualified.up(sin_i.absolute_upper() ** 2)
    z2_lo = qualified.down(cos_i.lo * cos_i.lo)
    z4_lo = qualified.down(z2_lo * z2_lo)
    z6_lo = qualified.down(z4_lo * z2_lo)
    fx2 = qualified.up(fx * fx)
    fy2 = qualified.up(fy * fy)
    first = qualified.up(2.0 * qualified.up(fx2 + fy2) / z4_lo)
    second = qualified.up(4.0 * max(fx2, fy2) * sin2 / z6_lo)
    return qualified.up(math.sqrt(qualified.up(first + second)))


def certify_bands(model: str, parameters: Sequence[float],
                  theta_max: float) -> list[tuple[float, float, float]]:
    """Return (theta_lo, theta_hi, pixel Hessian Frobenius upper bound) rows."""
    edges = band_edges(theta_max)
    bands = []
    for lo, hi in zip(edges, edges[1:]):
        if model == "OPENCV_FISHEYE":
            fx, fy, _cx, _cy, *coefficients = parameters
            hbar = qualified.band_bound(
                lo, hi, coefficients, focal_max=max(fx, fy),
                small_axis_limit=SMALL_AXIS_LIMIT, subdivisions=SUBDIVISIONS,
            )["pixel_frobenius_upper"]
        else:
            fx, fy, _cx, _cy = parameters
            hbar = pinhole_hbar(lo, hi, fx, fy)
        bands.append((lo, hi, hbar))
    return bands


# ----------------------------------------------------------------------------
# header I/O

_SELECTOR_MARKER = "static const PackedKbCameraSelector kPackedKbCameraSelectors[] = {"
_BLOB_PATTERN = re.compile(
    r"static const std::uint8_t kPackedKbTableBytes\[(\d+)\] = \{\n(.*?)\n\};", re.DOTALL)


@dataclass
class PackedTable:
    header_text: str
    selector_rows: list[str]
    blob: bytes

    @classmethod
    def load(cls, path: Path = HEADER_PATH) -> "PackedTable":
        text = Path(path).read_text(encoding="utf-8")
        start, end = _selector_bounds(text)
        rows = [line for line in text[start:end].splitlines() if line.strip()]
        match = _BLOB_PATTERN.search(text)
        if match is None:
            raise TableError("header lacks the packed byte array")
        blob = bytes(int(token, 16) for token in re.findall(r"0x([0-9a-fA-F]{2})", match.group(2)))
        if len(blob) != int(match.group(1)):
            raise TableError("packed byte-array length is inconsistent")
        _parse_blob(blob)
        return cls(text, rows, blob)

    def keys(self) -> set[tuple]:
        """Selector keys of rows that bind at unit scale (new rows always do)."""
        keys = set()
        for row in self.selector_rows:
            floats = [float.fromhex(token[:-1]) for token in re.findall(r"-?0x[0-9a-fA-F.]+p[+-]\d+f", row)]
            unsigned = [int(token) for token in re.findall(r"(?<![0-9A-Za-z_.])(\d+)u", row)]
            width, height, _index, model_id = unsigned[-4:]
            model = {1: "OPENCV_FISHEYE", 2: "PINHOLE"}[model_id]
            count = MODEL_PARAMETER_COUNTS[model]
            parameters, max_angle = floats[:count], floats[8]
            angle = 0 if model == "PINHOLE" else f32_word(max_angle)
            keys.add((model, width, height, tuple(f32_word(v) for v in parameters), angle))
        return keys

    def binds(self, camera: RuntimeCamera) -> bool:
        """Whether the renderer binds a row for ``camera``: the float32 rule of
        ``CertifiedRemainderTableOwner::bind``, native rows scaled by the
        raster ratio and unit-scale rows matched exactly."""
        import numpy as np

        f = np.float32
        active = [f(v) for v in camera.parameters]
        for row in self.selector_rows:
            floats = [f(float.fromhex(token[:-1])) for token in re.findall(r"-?0x[0-9a-fA-F.]+p[+-]\d+f", row)]
            unsigned = [int(token) for token in re.findall(r"(?<![0-9A-Za-z_.])(\d+)u", row)]
            native_w, native_h, _index, model_id = unsigned[-4:]
            unit = "kSelectorActiveCornerUnitScale" in row
            if model_id != MODEL_IDS[camera.model]:
                continue
            if unit and (camera.width != native_w or camera.height != native_h):
                continue
            x_scale = f(1.0) if unit else f(camera.width) / f(native_w)
            y_scale = f(1.0) if unit else f(camera.height) / f(native_h)
            expected = [floats[0] * x_scale, floats[1] * y_scale,
                        floats[2] if unit else floats[2] * x_scale, floats[3] if unit else floats[3] * y_scale]
            if active[:4] != expected:
                continue
            if camera.model == "OPENCV_FISHEYE" and (active[4:8] != floats[4:8] or f(camera.max_angle) != floats[8]):
                continue
            return True
        return False

    def append(self, cameras: Sequence[RuntimeCamera]) -> int:
        """Append rows for cameras that have none; returns the number added."""
        header = _parse_blob(self.blob)
        existing = self.keys()
        descriptors = self.blob[header["cameras_offset"]:header["bands_offset"]]
        bands = self.blob[header["bands_offset"]:]
        camera_count, band_count = header["camera_count"], header["band_count"]
        added = 0
        for camera in cameras:
            if camera.key() in existing:
                continue
            parameters = [f32(value) for value in camera.parameters]
            if camera.model == "PINHOLE":
                source_angle = pinhole_theta_max(camera.width, camera.height, parameters)
            else:
                source_angle = camera.max_angle
                if not 0.0 < source_angle <= math.pi:
                    raise TableError("fisheye maxAngle must lie in (0, pi]")
            theta_max = certificate_max_angle(source_angle, camera.model)
            rows = certify_bands(camera.model, parameters, theta_max)
            identity = hashlib.sha256(json.dumps(
                [camera.model, camera.width, camera.height, [float(v) for v in parameters]],
            ).encode("utf-8")).digest()
            descriptors += CAMERA.pack(identity, identity, identity, band_count, len(rows),
                                       MODEL_IDS[camera.model], camera.width, camera.height, 0,
                                       theta_max)
            bands += b"".join(BAND.pack(lo, hi, hbar) for lo, hi, hbar in rows)
            selector_parameters = (parameters + [0.0] * 8)[:8]
            self.selector_rows.append(
                "    {{" + ", ".join(float_literal(v) for v in selector_parameters) + "}, "
                f"{float_literal(theta_max)}, {camera.width}u, {camera.height}u, "
                f"{camera_count}u, {MODEL_IDS[camera.model]}u, kSelectorActiveCornerUnitScale}},")
            existing.add(camera.key())
            camera_count += 1
            band_count += len(rows)
            added += 1
        if added:
            bands_offset = HEADER_BYTES + camera_count * CAMERA_BYTES
            payload = bands_offset + band_count * BAND_BYTES
            packed = HEADER.pack(MAGIC, 2, ENDIAN_SENTINEL, HEADER_BYTES, CAMERA_BYTES, BAND_BYTES,
                                 camera_count, band_count, FLAGS, HEADER_BYTES, bands_offset,
                                 payload, header["digest0"], header["digest1"], header["digest2"])
            self.blob = packed + descriptors + bands
            _parse_blob(self.blob)
        return added

    def write(self, path: Path = HEADER_PATH) -> None:
        text = self.header_text
        start, end = _selector_bounds(text)
        text = text[:start] + "\n".join(self.selector_rows) + "\n" + text[end:]
        match = _BLOB_PATTERN.search(text)
        lines = [f"static const std::uint8_t kPackedKbTableBytes[{len(self.blob)}] = {{"]
        for offset in range(0, len(self.blob), 16):
            lines.append("    " + ", ".join(f"0x{b:02x}" for b in self.blob[offset:offset + 16]) + ",")
        lines.append("};")
        text = text[:match.start()] + "\n".join(lines) + text[match.end():]
        Path(path).write_text(text, encoding="utf-8")


def _selector_bounds(text: str) -> tuple[int, int]:
    at = text.find(_SELECTOR_MARKER)
    if at < 0:
        raise TableError("header lacks the camera selector array")
    start = text.find("\n", at) + 1
    end = text.find("};", start)
    return start, end


def _parse_blob(blob: bytes) -> dict:
    if len(blob) < HEADER_BYTES:
        raise TableError("packed table is shorter than its header")
    (magic, version, endian, header_bytes, camera_bytes, band_bytes, camera_count, band_count,
     flags, cameras_offset, bands_offset, payload_bytes, d0, d1, d2) = HEADER.unpack_from(blob)
    if (magic != MAGIC or version != 2 or endian != ENDIAN_SENTINEL
            or (header_bytes, camera_bytes, band_bytes, flags)
            != (HEADER_BYTES, CAMERA_BYTES, BAND_BYTES, FLAGS)
            or cameras_offset != HEADER_BYTES
            or bands_offset != cameras_offset + camera_count * CAMERA_BYTES
            or payload_bytes != bands_offset + band_count * BAND_BYTES
            or payload_bytes != len(blob)):
        raise TableError("packed table layout differs from the renderer ABI")
    return {"camera_count": camera_count, "band_count": band_count,
            "cameras_offset": cameras_offset, "bands_offset": bands_offset,
            "digest0": d0, "digest1": d1, "digest2": d2}


# ----------------------------------------------------------------------------
# cameras from COLMAP

def fisheye_max_angle(k: Sequence[float]) -> float:
    """Largest open angle on which rho(theta) is strictly increasing (or pi)."""
    k1, k2, k3, k4 = (float(v) for v in k)

    def derivative(theta: float) -> float:
        y = theta * theta
        return 1.0 + y * (3.0 * k1 + y * (5.0 * k2 + y * (7.0 * k3 + y * 9.0 * k4)))

    import numpy as np
    polynomial = np.asarray([9.0 * k4, 7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0], dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(polynomial) > 0.0)
    roots = np.roots(polynomial[int(nonzero[0]):])
    folds = sorted(math.sqrt(float(r.real)) for r in roots
                   if abs(float(r.imag)) <= 1.0e-10 * max(1.0, abs(float(r.real)))
                   and float(r.real) > 0.0 and math.sqrt(float(r.real)) < math.pi)
    limit = math.nextafter(folds[0] if folds else math.pi, 0.0)
    while derivative(limit) <= 0.0:
        limit = math.nextafter(limit, 0.0)
    return limit


def cameras_from_colmap(path: Path, width: int | None = None, height: int | None = None,
                        downsample: int = 1, max_angle: float | None = None) -> list[RuntimeCamera]:
    """Active cameras for a COLMAP cameras.txt/.bin under the corner pixel chart.

    Intrinsics are scaled by the exact per-axis raster ratio,
    f' = (W'/W) f and c' = (W'/W) c.
    """
    rows = _read_colmap_cameras(Path(path))
    result = []
    for model, native_w, native_h, params in rows:
        active_w = width if width is not None else native_w // downsample
        active_h = height if height is not None else native_h // downsample
        ax, ay = active_w / native_w, active_h / native_h
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            model, params = "PINHOLE", [f, f, cx, cy]
        if model not in MODEL_IDS:
            raise TableError(f"unsupported COLMAP camera model {model}")
        fx, fy, cx, cy = params[:4]
        active = [fx * ax, fy * ay, cx * ax, cy * ay, *params[4:8]]
        angle = 0.0
        if model == "OPENCV_FISHEYE":
            angle = max_angle if max_angle is not None else fisheye_max_angle(params[4:8])
        result.append(RuntimeCamera(model, active_w, active_h,
                                    tuple(f32(v) for v in active), angle))
    return result


_COLMAP_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 5: ("OPENCV_FISHEYE", 8)}


def _read_colmap_cameras(path: Path) -> list[tuple[str, int, int, list[float]]]:
    rows = []
    if path.suffix == ".bin":
        data = path.read_bytes()
        (count,), offset = struct.unpack_from("<Q", data, 0), 8
        for _ in range(count):
            _cid, model_id, w, h = struct.unpack_from("<iiQQ", data, offset)
            offset += 24
            if model_id not in _COLMAP_MODELS:
                raise TableError(f"unsupported COLMAP camera model id {model_id}")
            name, n = _COLMAP_MODELS[model_id]
            params = list(struct.unpack_from(f"<{n}d", data, offset))
            offset += 8 * n
            rows.append((name, int(w), int(h), params))
    else:
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            tokens = line.split()
            rows.append((tokens[1], int(tokens[2]), int(tokens[3]), [float(t) for t in tokens[4:]]))
    return rows


def dataset_cameras(dataset) -> list[tuple[int, RuntimeCamera | None, str]]:
    """``(camera id, runtime camera or None, model name)`` of every camera of a GOFish dataset, with the
    parameters the renderer receives; None marks a model the table cannot hold."""
    result = []
    for camera_id, (params, _origins, _directions, kind, _pixels) in dataset.intrinsics.items():
        width, height = (int(v) for v in params["resolution"])
        focal, centre = list(params["focal_length"]), list(params["principal_point"])
        if kind == "OpenCVFisheyeCameraModelParameters":
            camera = RuntimeCamera("OPENCV_FISHEYE", width, height,
                                   tuple(f32(v) for v in [*focal, *centre, *list(params["radial_coeffs"])[:4]]),
                                   float(params["max_angle"]))
            result.append((camera_id, camera, "OPENCV_FISHEYE"))
        elif kind == "OpenCVPinholeCameraModelParameters":
            distortion = [*params["radial_coeffs"], *params["tangential_coeffs"], *params["thin_prism_coeffs"]]
            if any(float(v) != 0.0 for v in distortion):
                result.append((camera_id, None, "distorted pinhole"))
            else:
                result.append((camera_id, RuntimeCamera("PINHOLE", width, height,
                                                        tuple(f32(v) for v in [*focal, *centre]), 0.0), "PINHOLE"))
        else:
            result.append((camera_id, None, kind))
    return result


def camera_argument(camera: RuntimeCamera) -> str:
    """``build_calibration_table --camera=`` value of ``camera``: comma-separated, floats in hex."""
    values = [camera.model, str(camera.width), str(camera.height), *(float(v).hex() for v in camera.parameters)]
    if camera.model == "OPENCV_FISHEYE":
        values.append(float(camera.max_angle).hex())
    return ",".join(values)


def check_dataset(dataset, path: Path = HEADER_PATH) -> None:
    """Raise with the fix when a camera of ``dataset`` has no complete-admission table row."""
    table = PackedTable.load(path)
    problems = []
    for camera_id, camera, model in dataset_cameras(dataset):
        if camera is None:
            problems.append(f"camera {camera_id}: {model} is not supported; undistort the images with COLMAP "
                            "(colmap image_undistorter) to PINHOLE")
        elif not table.binds(camera):
            problems.append(f"camera {camera_id}: no table row; add it, then start again:\n"
                            f"    python -m gofish.tools.build_calibration_table --camera={camera_argument(camera)}")
    if problems:
        raise TableError("complete admission needs a curvature-table row for every camera\n" + "\n".join(problems))
