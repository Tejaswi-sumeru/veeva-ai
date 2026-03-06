"""
Convert Figma-exported HTML (divs, flexbox, inline styles) to email-safe table-based HTML.
No dependencies beyond bs4 (already used elsewhere). Does not modify any existing app code.
"""

import io
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# Placeholder in generated HTML for image base URL; replaced with user's URL after generation
IMAGE_BASE_URL_PLACEHOLDER = "{{IMAGE_BASE_URL}}"

try:
    from bs4 import BeautifulSoup, NavigableString
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


def _parse_css_color(style_val: str) -> Optional[str]:
    """Extract hex color from style value; handle var(--name, #FFF) fallback."""
    if not style_val:
        return None
    # var(--surface-yellow, #FFD100) -> use fallback
    m = re.search(r"var\s*\(\s*[^,]+\s*,\s*(#[0-9A-Fa-f]{3,8})\s*\)", style_val)
    if m:
        return m.group(1)
    m = re.search(r"#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", style_val)
    if m:
        return "#" + m.group(1).upper() if len(m.group(1)) == 3 else "#" + m.group(1)
    return None


def _get_inline_style_dict(style_attr: Optional[str]) -> Dict[str, str]:
    """Parse inline style string into dict (lowercase keys)."""
    out: Dict[str, str] = {}
    if not style_attr or not isinstance(style_attr, str):
        return out
    for part in style_attr.split(";"):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            out[k.strip().lower()] = v.strip()
    return out


def _get_bgcolor_from_soup(soup: Any) -> str:
    """Get background color from root or first wrapper; default #FFFFFF."""
    if not soup:
        return "#FFFFFF"
    for el in soup.find_all(True):
        style = _get_inline_style_dict(el.get("style"))
        bg = style.get("background") or style.get("background-color")
        if bg:
            color = _parse_css_color(bg)
            if color:
                return color
    return "#FFFFFF"


def _text_and_style(el: Any, inherited: Optional[Dict[str, str]] = None) -> List[Tuple[str, Dict[str, str]]]:
    """Recursively collect (text, inline_style) from element. Inherited = parent computed style."""
    inherited = dict(inherited or {})
    style = _get_inline_style_dict(el.get("style"))
    # Merge: current wins
    for k, v in style.items():
        if k in ("color", "font-size", "font-family", "font-weight", "line-height", "text-align"):
            inherited[k] = v
    out: List[Tuple[str, Dict[str, str]]] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            t = child.strip()
            if t:
                out.append((t, dict(inherited)))
        else:
            out.extend(_text_and_style(child, inherited))
    return out


def _collect_blocks(soup: Any) -> List[Dict[str, Any]]:
    """
    Walk the tree and collect block-level content: text lines (with style), images, and implied spacers.
    Flattens div/flex structure into a linear list for table rows.
    """
    blocks: List[Dict[str, Any]] = []
    seen_text: set = set()

    def _walk(el: Any, inherited_style: Optional[Dict[str, str]] = None) -> None:
        if el is None:
            return
        style = _get_inline_style_dict(el.get("style"))
        merged = dict(inherited_style or {})
        for k, v in style.items():
            if k in ("color", "font-size", "font-family", "font-weight", "line-height", "text-align"):
                merged[k] = v

        if el.name == "img":
            src = el.get("src") or ""
            alt = el.get("alt") or "Image"
            w = el.get("width") or ""
            h = el.get("height") or ""
            if not w and style.get("width"):
                w = re.sub(r"[^\d]", "", style["width"]) or ""
            if not h and style.get("height"):
                h = re.sub(r"[^\d]", "", style["height"]) or ""
            blocks.append({"type": "image", "src": src, "alt": alt, "width": w, "height": h})
            return
        if el.name in ("script", "style", "head", "meta"):
            return

        # Check for direct text content (short circuit deep nesting)
        text_parts = _text_and_style(el, merged)
        if text_parts:
            full_text = " ".join(t for t, _ in text_parts)
            if full_text.strip() and full_text.strip() not in seen_text:
                seen_text.add(full_text.strip())
                # Build inline style string for email
                color = _parse_css_color(merged.get("color") or "") or "#222731"
                fs = merged.get("font-size") or "16px"
                ff = merged.get("font-family") or "Calibri, Arial, sans-serif"
                fw = merged.get("font-weight") or "400"
                blocks.append({
                    "type": "text",
                    "text": full_text,
                    "color": color,
                    "font_size": fs,
                    "font_family": ff,
                    "font_weight": fw,
                    "text_align": merged.get("text-align") or "center",
                })
            return

        for child in el.children:
            if hasattr(child, "name"):
                _walk(child, merged)
            elif isinstance(child, NavigableString) and child.strip():
                t = child.strip()
                if t not in seen_text:
                    seen_text.add(t)
                    blocks.append({
                        "type": "text",
                        "text": t,
                        "color": "#222731",
                        "font_size": "16px",
                        "font_family": "Calibri, Arial, sans-serif",
                        "font_weight": "400",
                        "text_align": "center",
                    })

    body = soup.find("body") or soup
    _walk(body)
    return blocks


def _px_to_num(s: str) -> int:
    """Parse 20px or 20 -> 20."""
    if not s:
        return 0
    s = re.sub(r"[^\d]", "", str(s))
    return int(s) if s else 0


def figma_html_to_email_tables(figma_html: str) -> str:
    """
    Convert Figma-exported HTML to email-safe table-based HTML.
    Returns a single string of table markup (no html/body wrapper).
    """
    if not BS4_AVAILABLE:
        return "<!-- BeautifulSoup required: pip install beautifulsoup4 -->"

    soup = BeautifulSoup(figma_html.strip(), "html.parser")
    bgcolor = _get_bgcolor_from_soup(soup)
    blocks = _collect_blocks(soup)

    def _row_spacer(height_px: int = 16) -> str:
        return (
            f'<tr><td align="center" height="{height_px}"></td></tr>'
        )

    def _wrap_section(inner: str, width_pct: str = "90%") -> str:
        return (
            f'<table bgcolor="{bgcolor}" border="0" cellpadding="0" cellspacing="0" role="presentation" '
            f'style="background-color:{bgcolor}; width:{width_pct};" width="{width_pct}">\n'
            f'  <tr><td align="center">\n{inner}\n  </td></tr>\n</table>'
        )

    lines: List[str] = []
    lines.append(
        f'<table bgcolor="{bgcolor}" border="0" cellpadding="0" cellspacing="0" role="presentation" '
        f'style="background-color:{bgcolor}; width:100%;" width="100%">'
    )
    lines.append("")
    lines.append("  <tr><td align=\"center\" height=\"40\"></td></tr>")
    lines.append("  <tr><td align=\"center\">")

    inner_parts: List[str] = []
    for i, blk in enumerate(blocks):
        if blk["type"] == "text":
            color = blk.get("color", "#222731")
            fs = blk.get("font_size", "20px")
            ff = blk.get("font_family", "Calibri, Arial, sans-serif")
            fw = blk.get("font_weight", "700")
            align = blk.get("text_align", "center")
            text = blk.get("text", "").replace("'", "&rsquo;").replace('"', "&quot;")
            inner_parts.append(
                _wrap_section(
                    f'<table bgcolor="{bgcolor}" border="0" cellpadding="0" cellspacing="0" role="presentation" '
                    f'style="background-color:{bgcolor}; width:90%;" width="90%">'
                    f'<tr><td class="f24" style="font-size:{fs}; text-align:{align}; font-weight:{fw}; '
                    f'font-family: {ff}; color:{color};">{text}</td></tr></table>'
                )
            )
            if i < len(blocks) - 1:
                inner_parts.append(_row_spacer(10))
        elif blk["type"] == "image":
            src = blk.get("src", "")
            alt = blk.get("alt", "Image")
            w = blk.get("width", "320")
            h = blk.get("height", "28")
            if not w or not re.match(r"^\d+$", str(w)):
                w = "320"
            if not h or not re.match(r"^\d+$", str(h)):
                h = "28"
            inner_parts.append(
                _wrap_section(
                    f'<img alt="{alt}" src="{src}" style="width: {w}px; padding: 0px; height: {h}px; text-align: center;" '
                    f'width="{w}" height="{h}">'
                )
            )
            if i < len(blocks) - 1:
                inner_parts.append(_row_spacer(20))

    lines.append("    " + "\n    ".join(inner_parts).replace("\n", "\n    "))
    lines.append("  </td></tr>")
    lines.append("  <tr><td align=\"center\" height=\"32\"></td></tr>")
    lines.append("</table>")
    return "\n".join(lines)


# ----- LLM-based conversion (Hugging Face) -----

SYSTEM_PROMPT = """You convert design HTML/CSS into email-safe HTML. Email clients only support tables and inline styles.

LAYOUT (mandatory):
- Emails are always 700px wide and center-aligned. Use this structure:
  1. Outer wrapper: <table bgcolor="HEX" border="0" cellpadding="0" cellspacing="0" role="presentation" style="background-color:HEX; width:100%;" width="100%">
  2. Each content block sits in: <tr><td align="center">...</td></tr>
  3. Main content lives in ONE inner table that is 700px wide and centered: <table border="0" cellpadding="0" cellspacing="0" role="presentation" style="width:700px; max-width:700px;" width="700">. Put this inside the <td align="center"> so the whole email is centered at 700px.
- Use spacer rows between sections: <tr><td align="center" height="20"></td></tr> (use 10, 16, 20, 32, 40 as needed).

RULES:
- Use ONLY <table>, <tr>, <td>, <a>, <img>. No <div>, <span>, flexbox, grid, or position.
- Every <table>: border="0" cellpadding="0" cellspacing="0" role="presentation"
- Colors: resolve CSS variables to hex (e.g. var(--surface-yellow, #FFD100) -> #FFD100). Use hex everywhere.

TEXT:
- Include every piece of text from the input: headings, subheadings, body copy, labels, button text. Do not omit or summarize.
- Apply CSS from the input to each text block: font-family (e.g. Arial, Calibri, helvetica, sans-serif), font-size (e.g. 20px, 32px), font-weight (bold/400), color (hex), text-align (center/left), line-height. Put all of these in the <td> inline style.

IMAGES:
- If an "Available images" section is provided, use those filenames and dimensions. Set src to {{IMAGE_BASE_URL}}/filename (e.g. {{IMAGE_BASE_URL}}/hero.png). Use the width and height listed for each image.
- Otherwise take width/height from the input HTML. Set BOTH attributes and style: <img src="..." alt="..." width="494" height="505" style="display:block; padding:0; width:494px; height:505px; border:0;">. Do not use arbitrary placeholder sizes.
- Wrap each image in <td align="center"> (and <a href="..."> if it is a link).

LINKS:
- Preserve href exactly. Use <a href="..." title="..."> around linked images or text.

OUTPUT:
- Start with <table and end with </table>. No markdown, no ```, no explanation. Raw HTML only.

EXAMPLE (700px centered, text with full inline CSS from design, image with explicit dimensions):
<table bgcolor="#FFD100" border="0" cellpadding="0" cellspacing="0" role="presentation" style="background-color:#FFD100; width:100%;" width="100%">
  <tr><td align="center" height="40"></td></tr>
  <tr><td align="center">
    <table bgcolor="#FFD100" border="0" cellpadding="0" cellspacing="0" role="presentation" style="background-color:#FFD100; width:700px; max-width:700px;" width="700">
      <tr><td align="center" style="font-size:20px; font-family: Arial, Calibri, helvetica, sans-serif; font-weight:bold; color:#222731;">Heading text here</td></tr>
    </table>
  </td></tr>
  <tr><td align="center" height="20"></td></tr>
  <tr><td align="center">
    <table border="0" cellpadding="0" cellspacing="0" role="presentation" style="width:700px; max-width:700px;" width="700">
      <tr><td align="center" style="font-size:32px; text-align:center; font-weight:bold; font-family: Arial, Calibri, sans-serif; color:#222731;">Body headline</td></tr>
    </table>
  </td></tr>
  <tr><td align="center" height="20"></td></tr>
  <tr><td align="center"><img alt="Hero" src="https://example.com/hero.png" width="494" height="505" style="display:block; padding:0; width:494px; height:505px; border:0;"></td></tr>
  <tr><td align="center" height="32"></td></tr>
</table>"""

USER_PROMPT_TEMPLATE = """Convert the HTML and CSS below into a single email-safe HTML fragment.

Requirements:
- Width 700px, center-aligned: one outer full-width table with background color, then all content in an inner table (or tables) with width="700" and style="width:700px; max-width:700px;" inside <td align="center">.
- Text: Include every heading, subheading, paragraph, and label from the input. Apply the CSS from the input to each: same font-family, font-size, color, font-weight, line-height, text-align as in the design (resolve var() to hex).
- Images: Use width and height from the input (inline style or element dimensions). Set both width/height attributes and style (display:block; width:Xpx; height:Ypx; border:0;). Keep src and alt exactly. Do not use wrong or placeholder dimensions.
- Output only valid HTML starting with <table and ending with </table>. No other text or markdown.

=== HTML ===
{html}

=== CSS (optional) ===
{css}

{images_section}
=== Email HTML (table only) ===
"""


def extract_images_from_zip(zip_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract image files from a ZIP. Returns list of {"filename": str, "width": int, "height": int}.
    Uses Pillow to get dimensions when possible; otherwise (0, 0).
    """
    result: List[Dict[str, Any]] = []
    allowed = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or "/." in name:
                continue
            base = name.split("/")[-1].lower()
            if not any(base.endswith(ext) for ext in allowed):
                continue
            width, height = 0, 0
            if Image:
                try:
                    with z.open(name) as f:
                        img = Image.open(f)
                        width, height = img.size
                except Exception:
                    pass
            result.append({"filename": name.split("/")[-1], "width": width, "height": height})
    return result


# FLAN-T5 has 512 token input limit; ~4 chars/token -> keep input under this many chars for local model
LOCAL_MODEL_MAX_INPUT_CHARS = 1200


def _build_prompt(
    html: str,
    css: str,
    image_list: Optional[List[Dict[str, Any]]] = None,
) -> str:
    css_block = css.strip() if css else "(none)"
    if image_list:
        lines = [
            "Available images (from your ZIP). Use these in <img> tags.",
            "For src use: " + IMAGE_BASE_URL_PLACEHOLDER + "/filename (e.g. " + IMAGE_BASE_URL_PLACEHOLDER + "/hero.png).",
            "Use the width and height below for each image.",
            "",
        ]
        for img in image_list:
            w, h = img.get("width", 0), img.get("height", 0)
            lines.append(f"- {img['filename']}  |  {w} x {h}")
        images_section = "=== Available images ===\n" + "\n".join(lines) + "\n\n"
    else:
        images_section = ""
    return USER_PROMPT_TEMPLATE.format(
        html=html.strip(),
        css=css_block,
        images_section=images_section,
    )


def _truncate_for_local_model(prompt: str) -> str:
    """Truncate prompt so it fits FLAN-T5 512-token limit; avoid indexing errors."""
    if len(prompt) <= LOCAL_MODEL_MAX_INPUT_CHARS:
        return prompt
    return prompt[:LOCAL_MODEL_MAX_INPUT_CHARS] + "\n\n[... input truncated for local model; use HF token for full HTML ...]"


def _call_hf_inference_api(prompt: str, token: Optional[str], model: str, max_new_tokens: int = 4096) -> str:
    """Use Hugging Face Inference API (serverless) for text generation."""
    try:
        from huggingface_hub import InferenceClient
        # First request can take 1–3 min while the model loads (cold start); allow up to 5 min
        client = InferenceClient(token=token or None, timeout=300.0)
        # Chat format works better for instruction following
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        out = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        if hasattr(out, "choices") and out.choices:
            c = out.choices[0]
            msg = getattr(c, "message", c)
            content = getattr(msg, "content", None) or getattr(msg, "text", "")
            return (content or "").strip()
        return ""
    except Exception as e:
        raise RuntimeError(f"Hugging Face API error: {e}") from e


def _call_local_pipeline(prompt: str, max_new_tokens: int = 1024) -> str:
    """Fallback: local FLAN-T5. Input is truncated to fit 512-token limit. On CPU, expect ~2–5 min."""
    prompt = _truncate_for_local_model(prompt)
    try:
        from transformers import pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1
        # FLAN-T5 max input length is 512; total (input+output) must fit
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_length=512,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(
            "Local model failed. Install: pip install transformers torch. "
            "Or set HF_TOKEN for Hugging Face API."
        ) from e
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nEmail HTML:"
    # FLAN-T5 accepts ~512 tokens total; ~4 chars/token -> cap at 2000 chars to be safe
    if len(full_prompt) > 2000:
        full_prompt = full_prompt[:2000] + "\n\n[truncated]"
    out = pipe(full_prompt, max_length=512, min_length=64)[0]
    return (out.get("generated_text") or "").strip()


def figma_html_to_email_with_llm(
    html: str,
    css: str = "",
    *,
    hf_token: Optional[str] = None,
    model: Optional[str] = None,
    use_local_fallback: bool = True,
    image_list: Optional[List[Dict[str, Any]]] = None,
    image_base_url: Optional[str] = None,
) -> str:
    """
    Convert Figma/design HTML (+ optional CSS) to email-safe table HTML using a Hugging Face model.
    If image_list is provided (e.g. from extract_images_from_zip), the prompt includes those
    images and the model should use {{IMAGE_BASE_URL}}/filename for src. image_base_url is
    then substituted in the output (if empty, placeholder is left for user to replace).
    """
    import os
    token = hf_token or os.environ.get("HF_TOKEN")
    prompt = _build_prompt(html, css or "", image_list=image_list)
    chosen_model = model or "mistralai/Mistral-7B-Instruct-v0.2"
    if token:
        raw = _call_hf_inference_api(prompt, token, chosen_model)
    elif use_local_fallback:
        raw = _call_local_pipeline(prompt)
    else:
        raise RuntimeError(
            "Set HF_TOKEN for Hugging Face Inference API, or enable local fallback "
            "(pip install transformers torch) and use_local_fallback=True."
        )
    # Strip markdown code fence if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    raw = raw.strip()
    # Replace image base URL placeholder so images point to user's server
    base = (image_base_url or "").rstrip("/")
    raw = raw.replace(IMAGE_BASE_URL_PLACEHOLDER, base if base else "https://yourserver.com/assets")
    return raw
