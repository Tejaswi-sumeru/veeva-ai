"""
Email image quality validation: classify by display size (icon / content / hero), then run quality checks only.
Quality = blur score (Laplacian) + compression ratio. No resolution/size/intrinsic checks.
Modular and usable standalone or from the main app.
"""
from __future__ import annotations

import base64
import io
import re
from typing import List, Optional, Tuple, Literal

# Display-size classification thresholds (px)
ICON_MAX_DIM = 150
CONTENT_MAX_DIM = 400

# Icon
ICON_MAX_BYTES = 150 * 1024
ICON_FORMATS = ("PNG", "JPEG", "GIF", "SVG")

# Content – quality only (blur + compression)
CONTENT_COMPRESSION_MIN = 0.00007
CONTENT_BLUR_MIN = 120

# Hero – quality only (blur + compression)
HERO_COMPRESSION_MIN = 0.00008
HERO_BLUR_MIN = 150
HERO_BRISQUE_MAX = 35  # optional; only if piq/BRISQUE available


def classify_image(display_w: int, display_h: int) -> Literal["icon", "content", "hero"]:
    """Classify by display size. Icon ≤150px, content 150–400px, hero >400px (max dimension)."""
    max_dim = max(display_w, display_h)
    if max_dim <= ICON_MAX_DIM:
        return "icon"
    if max_dim <= CONTENT_MAX_DIM:
        return "content"
    return "hero"


def get_image_bytes_from_src(src: str, timeout: int = 10) -> Optional[bytes]:
    """Fetch image bytes from data: URL or http(s) URL."""
    src = (src or "").strip()
    if not src:
        return None
    if src.startswith("data:"):
        idx = src.find("base64,")
        if idx == -1:
            return None
        try:
            return base64.b64decode(src[idx + 7 :])
        except Exception:
            return None
    if src.startswith("http://") or src.startswith("https://"):
        try:
            from urllib.request import urlopen
            with urlopen(src, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            return None
    return None


def _detect_format(image_bytes: bytes) -> str:
    """Detect format from magic bytes (for SVG and when PIL does not apply)."""
    if image_bytes.startswith(b"<?xml") or image_bytes.startswith(b"<svg") or image_bytes.lstrip().startswith(b"<svg"):
        return "SVG"
    if image_bytes[:8].startswith(b"\x89PNG"):
        return "PNG"
    if image_bytes[:2] == b"\xff\xd8":
        return "JPEG"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    return ""


def get_image_info(image_bytes: bytes) -> dict:
    """
    Return dict: format, width, height, file_size, pil_image (only for raster).
    For SVG, width/height may be None; pil_image is None.
    """
    file_size = len(image_bytes)
    fmt = _detect_format(image_bytes)
    width, height = None, None
    pil_image = None
    if fmt == "SVG":
        return {
            "format": fmt,
            "width": width,
            "height": height,
            "file_size": file_size,
            "pil_image": None,
        }
    try:
        from PIL import Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
        width, height = pil_image.size[0], pil_image.size[1]
        if not fmt:
            fmt = (pil_image.format or "").upper() or "UNKNOWN"
    except Exception:
        pass
    return {
        "format": fmt,
        "width": width,
        "height": height,
        "file_size": file_size,
        "pil_image": pil_image,
    }


def get_display_size_from_tag(tag) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse display width/height from a BeautifulSoup img tag (attributes or style).
    Returns (width, height) in pixels, or (None, None).
    """
    def parse_px(s: Optional[str]) -> Optional[int]:
        if not s or not isinstance(s, str):
            return None
        s = s.strip().rstrip("px;").strip()
        m = re.match(r"^(\d+)", s)
        return int(m.group(1)) if m else None

    w = parse_px(tag.get("width"))
    h = parse_px(tag.get("height"))
    if w is not None and h is not None:
        return w, h
    style = tag.get("style") or ""
    for part in style.split(";"):
        if "width" in part:
            w = w or parse_px(part.split(":")[-1])
        if "height" in part:
            h = h or parse_px(part.split(":")[-1])
    return w or None, h or None


def blur_score(pil_image) -> Optional[float]:
    """Laplacian variance (higher = sharper). Returns None if scipy/PIL unavailable."""
    if pil_image is None:
        return None
    try:
        import numpy as np
        from scipy.ndimage import laplace
        gray = np.asarray(pil_image.convert("L"), dtype=np.float64)
        lap = laplace(gray)
        return float(lap.var())
    except Exception:
        return None


def compression_ratio(file_size: int, width: Optional[int], height: Optional[int]) -> Optional[float]:
    """Bytes per pixel. Higher = less compressed. Returns None if dimensions missing."""
    if width is None or height is None or width * height == 0:
        return None
    return file_size / (width * height)


def validate_icon(
    info: dict,
    display_w: int,
    display_h: int,
) -> Tuple[bool, List[str]]:
    """Icon: format and file size only (no resolution/blur checks)."""
    errors: List[str] = []
    fmt = (info.get("format") or "").upper()
    file_size = info.get("file_size", 0)

    if fmt == "SVG":
        if file_size >= ICON_MAX_BYTES:
            errors.append(f"Icon file size {file_size // 1024} KB ≥ {ICON_MAX_BYTES // 1024} KB")
        return (len(errors) == 0, errors)

    if fmt and fmt not in ICON_FORMATS:
        errors.append(f"Icon format should be PNG/SVG/GIF (got {fmt})")
    if file_size >= ICON_MAX_BYTES:
        errors.append(f"Icon file size {file_size // 1024} KB ≥ {ICON_MAX_BYTES // 1024} KB")
    return (len(errors) == 0, errors)


def validate_content(
    info: dict,
    display_w: int,
    display_h: int,
) -> Tuple[bool, List[str]]:
    """Content: blur and compression only (no resolution/size checks)."""
    errors: List[str] = []
    w, h = info.get("width"), info.get("height")
    file_size = info.get("file_size", 0)
    pil = info.get("pil_image")

    if w is None or h is None:
        errors.append("Could not get dimensions (needed for quality checks)")
        return (False, errors)
    ratio = compression_ratio(file_size, w, h)
    if ratio is not None and ratio < CONTENT_COMPRESSION_MIN:
        errors.append(f"Content compression ratio {ratio:.6f} < {CONTENT_COMPRESSION_MIN}")
    blur = blur_score(pil)
    if blur is not None and blur < CONTENT_BLUR_MIN:
        errors.append(f"Content blur score {blur:.1f} < {CONTENT_BLUR_MIN} (too blurry)")
    elif pil is not None and blur is None:
        errors.append("Content blur score could not be computed")
    return (len(errors) == 0, errors)


def validate_hero(
    info: dict,
    display_w: int,
    display_h: int,
) -> Tuple[bool, List[str]]:
    """Hero: blur and compression only (no resolution/size checks)."""
    errors: List[str] = []
    w, h = info.get("width"), info.get("height")
    file_size = info.get("file_size", 0)
    pil = info.get("pil_image")

    if w is None or h is None:
        errors.append("Could not get dimensions (needed for quality checks)")
        return (False, errors)
    ratio = compression_ratio(file_size, w, h)
    if ratio is not None and ratio < HERO_COMPRESSION_MIN:
        errors.append(f"Hero compression ratio {ratio:.6f} < {HERO_COMPRESSION_MIN}")
    blur = blur_score(pil)
    if blur is not None and blur < HERO_BLUR_MIN:
        errors.append(f"Hero blur score {blur:.1f} < {HERO_BLUR_MIN} (too blurry)")
    elif pil is not None and blur is None:
        errors.append("Hero blur score could not be computed")
    return (len(errors) == 0, errors)


def validate_image(
    image_bytes: bytes,
    display_w: int,
    display_h: int,
    alt: str = "",
) -> Tuple[str, bool, List[str]]:
    """
    Classify by display size, run type-specific validation.
    Returns (role, ok, list of error messages).
    """
    role = classify_image(display_w, display_h)
    info = get_image_info(image_bytes)
    w, h = info.get("width"), info.get("height")

    if role == "icon":
        ok, errs = validate_icon(info, display_w, display_h)
    elif role == "content":
        ok, errs = validate_content(info, display_w, display_h)
    else:
        ok, errs = validate_hero(info, display_w, display_h)

    prefix = f"[{role}]"
    if alt:
        prefix = f"[{role}] Alt: {alt}"
    messages = [f"❌ {prefix} — {e}" for e in errs]
    return (role, ok, messages)


def validate_html_images(
    html_content: str,
    get_image_bytes_from_src_fn=None,
    get_display_size_fn=None,
    skip_src_pattern: str = "emltrk.com",
) -> List[str]:
    """
    Validate all images in HTML. Uses get_image_bytes_from_src and get_display_size_from_tag by default.
    Returns list of error strings (one per failing image or unverifiable).
    """
    results = validate_html_images_with_details(
        html_content, get_image_bytes_from_src_fn, get_display_size_fn, skip_src_pattern
    )
    return [r["message"] for r in results]


def validate_html_images_with_details(
    html_content: str,
    get_image_bytes_from_src_fn=None,
    get_display_size_fn=None,
    skip_src_pattern: str = "emltrk.com",
) -> List[dict]:
    """
    Like validate_html_images but returns list of dicts with keys: message, alt, image_bytes (optional).
    Use image_bytes to display a thumbnail of the failing image in the UI.
    """
    from bs4 import BeautifulSoup
    get_bytes = get_image_bytes_from_src_fn or get_image_bytes_from_src
    get_display = get_display_size_fn or get_display_size_from_tag
    details: List[dict] = []
    if not html_content:
        return details
    soup = BeautifulSoup(html_content, "html.parser")
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src or (skip_src_pattern and skip_src_pattern in src):
            continue
        alt = (img.get("alt") or "").strip() or "(no alt)"
        image_bytes = get_bytes(src)
        if not image_bytes:
            details.append({"message": f"⚠️ Could not load image (Alt: {alt})", "alt": alt, "image_bytes": None})
            continue
        display_w, display_h = get_display(img)
        if display_w is None or display_h is None:
            info = get_image_info(image_bytes)
            display_w = info.get("width") or 400
            display_h = info.get("height") or 400
        role, ok, msgs = validate_image(image_bytes, display_w, display_h, alt=alt)
        if not ok:
            for m in msgs:
                details.append({"message": m, "alt": alt, "image_bytes": image_bytes})
    return details
