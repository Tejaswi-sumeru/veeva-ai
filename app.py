"""
Streamlit UI for PDF Comparison

A simple web interface to upload two PDFs and view differences with highlighting.
"""

import streamlit as st
import tempfile
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from compare_pdfs import (
    PDFComparator,
    normalize_for_comparison,
)
import io
from html_processor import check_litmus_tracking, check_image_alt_matches_link_alias, check_missing_title_attributes

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PDF Comparison Tool",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'comparator' not in st.session_state:
    st.session_state.comparator = None
if 'comparison_done' not in st.session_state:
    st.session_state.comparison_done = False
if 'report' not in st.session_state:
    st.session_state.report = None
if 'text_diff' not in st.session_state:
    st.session_state.text_diff = None
if 'pdf1_path' not in st.session_state:
    st.session_state.pdf1_path = None
if 'pdf2_path' not in st.session_state:
    st.session_state.pdf2_path = None
if 'image_comparison' not in st.session_state:
    st.session_state.image_comparison = None
if 'font_comparison' not in st.session_state:
    st.session_state.font_comparison = None
if 'images1' not in st.session_state:
    st.session_state.images1 = None
if 'images2' not in st.session_state:
    st.session_state.images2 = None
if 'pdf_pages1' not in st.session_state:
    st.session_state.pdf_pages1 = None
if 'pdf_pages2' not in st.session_state:
    st.session_state.pdf_pages2 = None
if 'current_page_view' not in st.session_state:
    st.session_state.current_page_view = 1

def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory."""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Variable+state AMPscript collapse: show only blocks matching chosen variable state ---
#
# Parses %%[IF (@var == "value") THEN]%% (and ELSE/ENDIF). Stores (variable, state) per block.
# Emits TEXT only when the block's (variable, state) matches the chosen state for that variable.
# So for two IF blocks with same variable (@mfsAppDownloaded) but different states ("false"/"true"),
# only one box is shown (the one matching the chosen state). Default chosen state: "false".
#
RESOLVE_AMPSCRIPT = True


class Token(NamedTuple):
    type: str   # 'IF', 'ELSE', 'ENDIF', 'TEXT'
    value: str


AMP_PATTERN = re.compile(
    r"(%%\[\s*IF.*?THEN\s*\]%%|%%\[\s*ELSE\s*\]%%|%%\[\s*ENDIF\s*\]%%)",
    re.IGNORECASE | re.DOTALL,
)

# Match @VarName == "value" or @VarName == 'value' or @VarName == value (unquoted)
_COND_VAR_PATTERN = re.compile(
    r"@([A-Za-z0-9_]+)\s*==\s*(?:[\"']([^\"']*)[\"']|(\S+))",
    re.IGNORECASE,
)


def parse_if_condition(if_tag: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (variable, state) from %%[IF (@var == "value") THEN]%%.
    Returns (variable_key, state) e.g. ("mfsAppDownloaded", "false"), or (None, None) if unparseable.
    """
    m = _COND_VAR_PATTERN.search(if_tag)
    if not m:
        return (None, None)
    var_name = (m.group(1) or "").strip().lower()
    state_quoted = m.group(2)
    state_unquoted = m.group(3)
    state = (state_quoted or state_unquoted or "").strip()
    if not var_name or not state:
        return (None, None)
    return (var_name, state)


def _flip_boolean_state(state: str) -> str:
    """Return the opposite state for boolean-like values (for ELSE branch)."""
    s = state.strip().lower()
    if s == "false":
        return "true"
    if s == "true":
        return "false"
    if s in ("0", "1"):
        return "1" if s == "0" else "0"
    return state  # unknown, keep as-is


def tokenize_ampscript(html: str) -> List[Token]:
    """Split HTML into AMPscript tokens (IF/ELSE/ENDIF) and TEXT."""
    parts = AMP_PATTERN.split(html)
    tokens: List[Token] = []
    for part in parts:
        if not part:
            continue
        p_upper = part.upper().strip()
        if p_upper.startswith("%%[") and "IF" in p_upper and "THEN" in p_upper:
            tokens.append(Token("IF", part))
        elif p_upper.startswith("%%[") and "ELSE" in p_upper and "ENDIF" not in p_upper:
            tokens.append(Token("ELSE", part))
        elif p_upper.startswith("%%[") and "ENDIF" in p_upper:
            tokens.append(Token("ENDIF", part))
        else:
            tokens.append(Token("TEXT", part))
    return tokens


def get_ampscript_variables(html_content: str) -> Dict[str, List[str]]:
    """
    Scan HTML for %%[IF (@var == "value") THEN]%% (and ELSE) and return
    variable name -> list of possible states. Used to build variable-state UI.
    """
    html = (html_content or "").strip()
    tokens = tokenize_ampscript(html)
    seen_states_by_var: Dict[str, set] = {}
    stack: List[Tuple[Optional[str], Optional[str]]] = []

    for token in tokens:
        if token.type == "IF":
            var, state = parse_if_condition(token.value)
            stack.append((var, state))
            if var and state:
                seen_states_by_var.setdefault(var, set()).add(state)
        elif token.type == "ELSE":
            if stack:
                var, state = stack[-1]
                if var is not None and state is not None:
                    other = _flip_boolean_state(state)
                    stack[-1] = (var, other)
                    seen_states_by_var.setdefault(var, set()).add(other)
        elif token.type == "ENDIF":
            if stack:
                stack.pop()

    return {var: sorted(states) for var, states in seen_states_by_var.items()}

def resolve_and_strip_ampscript(
    html_content: str,
    chosen_state: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve AMPscript by variable+state: emit only blocks whose (variable, state) matches
    chosen_state for that variable. Blocks that don't match are removed.
    chosen_state: e.g. {"mfsappdownloaded": "false"}. If None, built from HTML (default "false"
    when both "false" and "true" appear for a variable).
    """
    html = (html_content or "").strip()
    if not RESOLVE_AMPSCRIPT:
        resolved = re.sub(r"%%\[[\s\S]*?\]%%", "", html)
        resolved = re.sub(r"\n\s+\n", "\n\n", resolved)
        return resolved.strip()

    tokens = tokenize_ampscript(html)
    # First pass: collect all (var, state) from IF tokens to build default chosen_state
    seen_states_by_var: Dict[str, set] = {}
    for token in tokens:
        if token.type == "IF":
            var, state = parse_if_condition(token.value)
            if var and state:
                seen_states_by_var.setdefault(var, set()).add(state)

    if chosen_state is None:
        chosen_state = {}
        for var, states in seen_states_by_var.items():
            # var is already lowercased from parse_if_condition
            false_match = next((s for s in states if s.lower() == "false"), None)
            if false_match is not None:
                chosen_state[var] = false_match
            else:
                chosen_state[var] = next(iter(states))

    output: List[str] = []
    stack: List[Tuple[Optional[str], Optional[str]]] = []  # (variable, state) per block

    for token in tokens:
        if token.type == "IF":
            var, state = parse_if_condition(token.value)
            stack.append((var, state))
        elif token.type == "ELSE":
            if stack:
                var, state = stack[-1]
                if var is not None and state is not None:
                    other = _flip_boolean_state(state)
                    stack[-1] = (var, other)
                # else leave unparseable block as-is (we'll suppress TEXT for None var)
        elif token.type == "ENDIF":
            if stack:
                stack.pop()
        else:  # TEXT
            if not stack:
                output.append(token.value)
            else:
                var, state = stack[-1]
                if var is None or state is None:
                    # Unparseable block: suppress to avoid leaking both branches
                    continue
                if chosen_state.get(var, "").lower() == state.lower():
                    output.append(token.value)

    if stack:
        raise ValueError("Unbalanced AMPscript IF/ENDIF detected")

    resolved = "".join(output)
    resolved = re.sub(r"%%\[[\s\S]*?\]%%", "", resolved)
    resolved = re.sub(r"\n\s+\n", "\n\n", resolved)
    return resolved.strip()


def html_to_pdf(html_content: str, output_path: str, chosen_state: Optional[Dict[str, str]] = None,
                page_width: float = None, page_height: float = None) -> bool:
    """
    Convert HTML content to PDF using Playwright (headless browser).
    Resolves AMPscript %%[IF (@var == "value") THEN]%% by variable state (chosen_state),
    then strips remaining %%[...]%% directives before conversion.
    """
    try:
        from playwright.sync_api import sync_playwright
        import re

        try:
            html_processed = resolve_and_strip_ampscript(html_content, chosen_state=chosen_state)
        except ValueError as e:
            st.error(f"❌ AMPscript error: {e}")
            return False

        if not re.search(r'<html[^>]*>', html_processed, re.IGNORECASE):
            html_processed = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }}
        h1, h2, h3, h4, h5, h6 {{ margin-top: 1em; margin-bottom: 0.5em; }}
        p {{ margin: 0.5em 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_processed}
</body>
</html>"""
        
        # Use Playwright to convert HTML to PDF
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception:
                import subprocess
                import sys
                st.warning("⚠️ Playwright Chromium not installed. Installing now (first time on this environment, may take 1–2 minutes)...")
                install_placeholder = st.empty()
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "playwright", "install", "chromium"],
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if result.returncode != 0:
                        install_placeholder.error(
                            f"❌ Playwright install failed. On Streamlit Cloud, ensure `packages.txt` exists with Chromium deps. "
                            f"Details: {result.stderr or result.stdout or 'unknown'}"
                        )
                        return False
                    install_placeholder.success("✅ Chromium installed. Launching...")
                    browser = p.chromium.launch(headless=True)
                except subprocess.TimeoutExpired:
                    install_placeholder.error("❌ Install timed out (e.g. on Streamlit Cloud). Try again or run locally.")
                    return False
                except Exception as install_error:
                    install_placeholder.error(f"❌ Failed to install/launch Chromium: {str(install_error)}")
                    st.info("On Streamlit Cloud: add a `packages.txt` with Chromium system deps (see repo).")
                    return False
            
            try:
                page = browser.new_page()
                
                # Use provided page size or calculate dynamically
                if page_width and page_height:
                    # Use the provided page size (matching PDF1)
                    inch_w = page_width
                    inch_h = page_height
                    
                    # Set viewport to match the target page size (convert inches to pixels at 96 DPI)
                    viewport_width = int(page_width * 96)
                    viewport_height = int(page_height * 96)
                    page.set_viewport_size({"width": viewport_width, "height": viewport_height})
                    
                    # Log the dimensions being used
                    st.info(f"📐 Matching PDF1 page size: {inch_w:.2f}\" × {inch_h:.2f}\" ({viewport_width}px × {viewport_height}px)")
                else:
                    # Dynamic sizing (existing logic)
                    page.set_content(html_processed, wait_until="networkidle")
                    dims = page.evaluate("""() => {
                        const el = document.documentElement;
                        const body = document.body;
                        const h = Math.max(el.scrollHeight, el.offsetHeight, body.scrollHeight, body.offsetHeight);
                        const w = Math.max(el.scrollWidth, el.offsetWidth, body.scrollWidth, body.offsetWidth);
                        return { width: w, height: h };
                    }""")
                    px_w = dims.get("width", 816)
                    px_h = dims.get("height", 1056)
                    inch_w = max(8.5, px_w / 96.0 + 0.5)
                    inch_h = max(11.0, px_h / 96.0 + 0.5)
                
                # Set content after viewport is configured
                if page_width and page_height:
                    page.set_content(html_processed, wait_until="networkidle")
                
                page.pdf(
                    path=output_path,
                    width=f"{inch_w}in",
                    height=f"{inch_h}in",
                    print_background=True,
                    prefer_css_page_size=False,  # Force content to fit specified dimensions
                    margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"},
                    scale=1.0  # Ensure no scaling is applied
                )
                browser.close()
                return True
                
            except Exception as e:
                browser.close()
                raise e
        
    except ImportError:
        st.error("❌ Playwright not installed. Install with: pip install playwright")
        st.info("After installing, run: playwright install chromium")
        return False
    except Exception as e:
        st.error(f"❌ Failed to convert HTML to PDF: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language='python')
        return False

def highlight_pdf_removals(pdf1_path, text_diff, output_path):
    """
    Create a highlighted version of PDF1 showing removed content.
    Works with chunk-based comparison.
    """
    try:
        import fitz  # PyMuPDF
        import re
    except ImportError:
        st.error("PyMuPDF (fitz) is required for PDF highlighting. Install with: pip install PyMuPDF")
        return False
    
    try:
        doc = fitz.open(pdf1_path)
        removed_chunks = text_diff.get('removed_chunks', [])
        highlighted_count = 0
        used_rects = set() # Track already highlighted areas to avoid overlap
        
        for chunk in removed_chunks:
            text_to_find = chunk.strip()
            if len(text_to_find) < 10: continue # Skip very short snippets to avoid false positives
            
            found = False
            for page_num in range(len(doc)):
                if found: break
                page = doc[page_num]
                # Search for the specific text
                instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
                
                for inst in instances:
                    # Check if this area is already highlighted
                    rect_key = (page_num, round(inst.x0, 1), round(inst.y0, 1))
                    if rect_key in used_rects: continue
                    
                    try:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1.0, 0, 0)) # Red
                        highlight.set_opacity(0.3)
                        highlight.update()
                        highlighted_count += 1
                        used_rects.add(rect_key)
                        found = True
                        break # Only highlight the first instance found for this chunk
                    except: pass
            
            # Fallback for large chunks (split into sentences)
            if not found and len(text_to_find) > 40:
                for sub in re.split(r'[.!?]+\s+', text_to_find):
                    if len(sub.strip()) < 15: continue
                    sub_found = False
                    for page_num in range(len(doc)):
                        if sub_found: break
                        page = doc[page_num]
                        for inst in page.search_for(sub.strip(), flags=fitz.TEXT_DEHYPHENATE):
                            rect_key = (page_num, round(inst.x0, 1), round(inst.y0, 1))
                            if rect_key in used_rects: continue
                            try:
                                highlight = page.add_highlight_annot(inst)
                                highlight.set_colors(stroke=(1.0, 0, 0))
                                highlight.update()
                                highlighted_count += 1
                                used_rects.add(rect_key)
                                sub_found = True
                                break
                            except: pass
        
        doc.save(output_path)
        doc.close()
        return True
    except Exception as e:
        st.error(f"Highlight removal error: {str(e)}")
        return False

def export_highlighted_html_to_pdf(highlighted_html: str, page_width: float = 8.5, page_height: float = 11.0) -> bytes:
    """
    Renders highlighted HTML directly to PDF bytes using Playwright.
    Preserves all injected <span> highlights and styles.
    """
    from playwright.sync_api import sync_playwright
    import tempfile
    import os

    # Wrap in full HTML boilerplate if missing
    if "<html" not in highlighted_html.lower():
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; padding: 40px; line-height: 1.5; }}
        .diff-added {{ background-color: #d4edda !important; border-bottom: 2px solid #28a745 !important; padding: 2px; border-radius: 2px; -webkit-print-color-adjust: exact; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {highlighted_html}
</body>
</html>"""
    else:
        # Inject our highlight styles if not present
        if ".diff-added" not in highlighted_html:
            style_tag = """<style>.diff-added { background-color: #d4edda !important; border-bottom: 2px solid #28a745 !important; padding: 2px; border-radius: 2px; -webkit-print-color-adjust: exact; }</style>"""
            if "</head>" in highlighted_html:
                full_html = highlighted_html.replace("</head>", f"{style_tag}</head>")
            else:
                full_html = style_tag + highlighted_html
        else:
            full_html = highlighted_html

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Match PDF1 dimensions
        vw = int(page_width * 96)
        vh = int(page_height * 96)
        page.set_viewport_size({"width": vw, "height": vh})
        
        page.set_content(full_html, wait_until="networkidle")
        
        pdf_bytes = page.pdf(
            width=f"{page_width}in",
            height=f"{page_height}in",
            print_background=True,
            margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"}
        )
        browser.close()
        return pdf_bytes

def highlight_pdf_differences(pdf_path, text_diff, output_path):
    """
    Create a highlighted version of a PDF showing additions and removals.
    Args:
        pdf_path: Path to the PDF to highlight.
        text_diff: Dictionary containing 'added_chunks' and/or 'removed_chunks'.
        output_path: Where to save the highlighted PDF.
    """
    try:
        import fitz
        import re
    except ImportError:
        return False
    
    try:
        doc = fitz.open(pdf_path)
        
        # Define types to process: (key, color_name, stroke_color)
        to_process = [
            ('added_chunks', 'Green', (0.0, 1.0, 0.0)),
            ('removed_chunks', 'Red', (1.0, 0.0, 0.0))
        ]
        
        highlighted_count = 0
        used_rects = set()
        
        for key, color_name, stroke_color in to_process:
            chunks = text_diff.get(key, [])
            if not chunks: continue
            
            print(f"[DEBUG] Highlighting {len(chunks)} {color_name} chunks in {pdf_path}")
            
            for chunk in chunks:
                text_to_find = chunk.strip()
                if len(text_to_find) < 10: continue
                
                found = False
                for page_num in range(len(doc)):
                    if found: break
                    page = doc[page_num]
                    instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
                    for inst in instances:
                        rect_key = (page_num, round(inst.x0, 1), round(inst.y0, 1))
                        if rect_key in used_rects: continue
                        
                        try:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=stroke_color) 
                            highlight.set_opacity(0.4)
                            highlight.update()
                            highlighted_count += 1
                            used_rects.add(rect_key)
                            found = True
                            break 
                        except: pass
                
                # Fallback to sentence search if whole chunk not found
                if not found and len(text_to_find) > 40:
                    for sub in re.split(r'[.!?]+\s+', text_to_find):
                        if len(sub.strip()) < 15: continue
                        sub_found = False
                        for page_num in range(len(doc)):
                            if sub_found: break
                            page = doc[page_num]
                            for inst in page.search_for(sub.strip(), flags=fitz.TEXT_DEHYPHENATE):
                                rect_key = (page_num, round(inst.x0, 1), round(inst.y0, 1))
                                if rect_key in used_rects: continue
                                try:
                                    highlight = page.add_highlight_annot(inst)
                                    highlight.set_colors(stroke=stroke_color)
                                    highlight.set_opacity(0.4)
                                    highlight.update()
                                    highlighted_count += 1
                                    used_rects.add(rect_key)
                                    sub_found = True
                                    break
                                except: pass
                            
        doc.save(output_path)
        doc.close()
        print(f"[DEBUG] Highlighted {highlighted_count} areas in total.")
        return True
    except Exception as e:
        print(f"[DEBUG] Highlight error: {str(e)}")
        st.error(f"Highlight error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Highlight addition error: {str(e)}")
        return False

def validate_pdf_checkpoints(pdf_path, checkpoints, config):
    """
    Validate PDF against selected checkpoints.
    
    Args:
        pdf_path: Path to PDF file
        checkpoints: Dictionary of selected checkpoints
        config: Configuration for each checkpoint
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {k: {'status': 'error', 'message': 'PyMuPDF not installed'} for k in checkpoints.keys() if checkpoints[k]}
    
    try:
        doc = fitz.open(pdf_path)
        
        # Font Check
        if checkpoints.get('font_arial', False):
            try:
                required_font = config.get('font', 'Arial').lower().strip()
                unique_fonts = set()
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    r = page.rect
                    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                    for clip in (
                        fitz.Rect(r.x0, r.y0, cx, cy),
                        fitz.Rect(cx, r.y0, r.x1, cy),
                        fitz.Rect(r.x0, cy, cx, r.y1),
                        fitz.Rect(cx, cy, r.x1, r.y1),
                    ):
                        try:
                            text_dict = page.get_text("dict", clip=clip)
                            for block in text_dict.get("blocks", []):
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        fn = span.get("font", "") or ""
                                        if not fn or fn == "Unknown":
                                            continue
                                        if "+" in fn:
                                            fn = fn.split("+")[-1]
                                        fn = fn.replace("CIDFont+", "").replace("TrueType+", "")
                                        fn = re.sub(r"^[A-Z0-9]+[+-]", "", fn).strip()
                                        if fn and fn.lower() != "unknown":
                                            unique_fonts.add(fn.lower())
                        except Exception:
                            continue
                    try:
                        font_list = page.get_fonts(full=True)
                        for fi in font_list:
                            name = base = ""
                            if isinstance(fi, dict):
                                name, base = fi.get("name", ""), fi.get("basefont", "")
                            elif isinstance(fi, (tuple, list)) and len(fi) >= 5:
                                base = fi[3] if len(fi) > 3 else ""
                                name = fi[4] if len(fi) > 4 else ""
                            font_to_use = (base or name or "").strip()
                            if not font_to_use or font_to_use == "Unknown":
                                continue
                            if "+" in font_to_use:
                                font_to_use = font_to_use.split("+")[-1]
                            font_to_use = font_to_use.replace("CIDFont+", "").replace("TrueType+", "")
                            font_to_use = re.sub(r"^[A-Z0-9]+[+-]", "", font_to_use).strip()
                            if font_to_use and font_to_use.lower() != "unknown":
                                unique_fonts.add(font_to_use.lower())
                    except Exception:
                        pass
                if st.session_state.comparator:
                    try:
                        fonts_info = st.session_state.comparator.extract_fonts_from_pdf(str(pdf_path))
                        uf = fonts_info.get("unique_fonts") or set()
                        if isinstance(uf, (list, tuple)):
                            uf = set(f for f in uf if f)
                        unique_fonts.update(uf)
                    except Exception:
                        pass
                def _norm(s):
                    return re.sub(r"[^a-z]", "", (s or "").lower())
                req_norm = _norm(required_font)
                font_found = any(
                    req_norm in _norm(f) or _norm(f) in req_norm
                    for f in unique_fonts
                )
                if font_found:
                    results['font_arial'] = {
                        'status': 'pass',
                        'message': f'Font "{config.get("font", "Arial")}" found in PDF',
                        'details': {
                            'Required font': config.get('font', 'Arial'),
                            'Found fonts': ', '.join(sorted(unique_fonts)) if unique_fonts else 'None detected'
                        }
                    }
                else:
                    results['font_arial'] = {
                        'status': 'fail',
                        'message': f'Font "{config.get("font", "Arial")}" not found in PDF',
                        'details': {
                            'Required font': config.get('font', 'Arial'),
                            'Found fonts': ', '.join(sorted(unique_fonts)) if unique_fonts else 'None detected'
                        }
                    }
            except Exception as e:
                results['font_arial'] = {
                    'status': 'error',
                    'message': f'Font check failed: {str(e)}',
                    'details': {}
                }
        
        # Logo Check
        if checkpoints.get('logo_check', False):
            logo_path = config.get('logo_path')
            if not logo_path:
                results['logo_check'] = {
                    'status': 'error',
                    'message': 'No reference logo uploaded',
                    'details': {}
                }
            else:
                try:
                    from PIL import Image
                    import imagehash
                    
                    ref_logo = Image.open(logo_path)
                    ref_hash = imagehash.phash(ref_logo)
                    ref_w, ref_h = ref_logo.size
                    min_w = max(30, int(ref_w * 0.25))
                    min_h = max(30, int(ref_h * 0.25))
                    logo_similarity_threshold = 0.90
                    logo_found = False
                    best_match = None
                    best_similarity = 0
                    pdf_images = []
                    mat = fitz.Matrix(1.5, 1.5)
                    max_pages_scan = min(15, len(doc))
                    step_x = max(8, ref_w // 2)
                    step_y = max(8, ref_h // 2)
                    
                    if st.session_state.comparator:
                        pdf_images = st.session_state.comparator.extract_images_from_pdf(str(pdf_path))
                        for img_data in pdf_images:
                            if 'pil_image' not in img_data:
                                continue
                            w = img_data.get('width', 0)
                            h = img_data.get('height', 0)
                            if w < min_w or h < min_h:
                                continue
                            img_hash = imagehash.phash(img_data['pil_image'])
                            similarity = 1 - (ref_hash - img_hash) / 256.0
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = img_data
                            if similarity >= logo_similarity_threshold:
                                logo_found = True
                                break
                    
                    if not logo_found and PIL_AVAILABLE:
                        for page_num in range(max_pages_scan):
                            if logo_found:
                                break
                            page = doc[page_num]
                            r = page.rect
                            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                            for clip in (
                                fitz.Rect(r.x0, r.y0, cx, cy),
                                fitz.Rect(cx, r.y0, r.x1, cy),
                                fitz.Rect(r.x0, cy, cx, r.y1),
                                fitz.Rect(cx, cy, r.x1, r.y1),
                            ):
                                if logo_found:
                                    break
                                try:
                                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                                    if pix.width < ref_w or pix.height < ref_h:
                                        continue
                                    region = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                    for y in range(0, region.height - ref_h + 1, step_y):
                                        if logo_found:
                                            break
                                        for x in range(0, region.width - ref_w + 1, step_x):
                                            crop = region.crop((x, y, x + ref_w, y + ref_h))
                                            ch = imagehash.phash(crop)
                                            sim = 1 - (ref_hash - ch) / 256.0
                                            if sim > best_similarity:
                                                best_similarity = sim
                                                best_match = {'page': page_num + 1}
                                            if sim >= logo_similarity_threshold:
                                                logo_found = True
                                                best_match = best_match or {'page': page_num + 1}
                                                break
                                except Exception:
                                    continue
                    
                    if logo_found:
                        results['logo_check'] = {
                            'status': 'pass',
                            'message': 'Reference logo found in PDF',
                            'details': {
                                'Best match similarity': f'{best_similarity * 100:.1f}%',
                                'Match location': f"Page {best_match.get('page', 'N/A')}" if best_match else 'N/A'
                            }
                        }
                    else:
                        results['logo_check'] = {
                            'status': 'fail',
                            'message': 'Reference logo not found in PDF',
                            'details': {
                                'Best match similarity': f'{best_similarity * 100:.1f}%' if best_match else '0%',
                                'Total images checked': len(pdf_images)
                            }
                        }
                except Exception as e:
                    results['logo_check'] = {
                        'status': 'error',
                        'message': f'Error checking logo: {str(e)}',
                        'details': {}
                    }
        
        # Color Check
        if checkpoints.get('color_check', False):
            required_colors = config.get('colors', [])
            if not required_colors:
                results['color_check'] = {
                    'status': 'error',
                    'message': 'No color codes specified',
                    'details': {}
                }
            elif not PIL_AVAILABLE:
                results['color_check'] = {
                    'status': 'error',
                    'message': 'PIL (Pillow) is required for color check. Install with: pip install Pillow',
                    'details': {}
                }
            else:
                try:
                    # Convert hex to RGB
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    
                    required_rgbs = [hex_to_rgb(c) for c in required_colors]
                    
                    # Extract colors from PDF (sample from rendered pages)
                    found_colors = set()
                    color_tolerance = 10  # RGB tolerance for color matching
                    
                    # Sample colors from first few pages
                    for page_num in range(min(3, len(doc))):
                        page = doc[page_num]
                        # Get page as image (alpha=False for RGB bytes)
                        pix = page.get_pixmap(alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Sample colors (every 100th pixel for performance)
                        for y in range(0, img.height, 100):
                            for x in range(0, img.width, 100):
                                r, g, b = img.getpixel((x, y))
                                found_colors.add((r, g, b))
                    
                    # Check if required colors are found
                    matched_colors = []
                    for req_color, req_rgb in zip(required_colors, required_rgbs):
                        found = False
                        for found_rgb in found_colors:
                            # Check if colors match within tolerance
                            if all(abs(found_rgb[i] - req_rgb[i]) <= color_tolerance for i in range(3)):
                                matched_colors.append(req_color)
                                found = True
                                break
                        if not found:
                            matched_colors.append(None)
                    
                    all_matched = all(c is not None for c in matched_colors)
                    
                    if all_matched:
                        results['color_check'] = {
                            'status': 'pass',
                            'message': f'All required colors found in PDF',
                            'details': {
                                'Required colors': ', '.join(required_colors),
                                'Matched colors': ', '.join([c for c in matched_colors if c])
                            }
                        }
                    else:
                        missing = [c for c, m in zip(required_colors, matched_colors) if m is None]
                        results['color_check'] = {
                            'status': 'fail',
                            'message': f'Some required colors not found: {", ".join(missing)}',
                            'details': {
                                'Required colors': ', '.join(required_colors),
                                'Found colors': ', '.join([c for c in matched_colors if c]),
                                'Missing colors': ', '.join(missing)
                            }
                        }
                except Exception as e:
                    results['color_check'] = {
                        'status': 'error',
                        'message': f'Error checking colors: {str(e)}',
                        'details': {}
                    }
        
        # Page Count Check
        if checkpoints.get('page_count', False):
            expected = config.get('page_count', 1)
            actual = len(doc)
            
            if actual == expected:
                results['page_count'] = {
                    'status': 'pass',
                    'message': f'Page count matches: {actual} pages',
                    'details': {
                        'Expected': expected,
                        'Actual': actual
                    }
                }
            else:
                results['page_count'] = {
                    'status': 'fail',
                    'message': f'Page count mismatch: expected {expected}, found {actual}',
                    'details': {
                        'Expected': expected,
                        'Actual': actual,
                        'Difference': actual - expected
                    }
                }
        
        # Text Content Check (supports multiple phrases separated by commas)
        if checkpoints.get('text_content', False):
            required_text = config.get('text', '').strip()
            if not required_text:
                results['text_content'] = {
                    'status': 'error',
                    'message': 'No text specified for check',
                    'details': {}
                }
            else:
                # Split by comma to allow multiple phrases; strip each
                phrases = [p.strip() for p in required_text.split(',') if p.strip()]
                if not phrases:
                    results['text_content'] = {
                        'status': 'error',
                        'message': 'No text specified for check',
                        'details': {}
                    }
                else:
                    # Extract all text from PDF
                    full_text = ""
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        full_text += page.get_text() + "\n"
                    full_text_lower = full_text.lower()
                    
                    found_phrases = []
                    missing_phrases = []
                    all_occurrences = {}
                    for phrase in phrases:
                        if phrase.lower() in full_text_lower:
                            found_phrases.append(phrase)
                            # Find occurrences for this phrase
                            occurrences = []
                            text_lower = full_text_lower
                            search_lower = phrase.lower()
                            start = 0
                            while True:
                                pos = text_lower.find(search_lower, start)
                                if pos == -1:
                                    break
                                page_num = 0
                                char_count = 0
                                for p in range(len(doc)):
                                    page_text = doc[p].get_text()
                                    if char_count + len(page_text) > pos:
                                        page_num = p
                                        break
                                    char_count += len(page_text)
                                occurrences.append(f"Page {page_num + 1}")
                                start = pos + 1
                            all_occurrences[phrase[:50]] = occurrences
                        else:
                            missing_phrases.append(phrase)
                    
                    all_found = len(missing_phrases) == 0
                    if all_found:
                        occ_lines = [f'"{k}": {len(v)} time(s) on {", ".join(v[:3])}{"..." if len(v) > 3 else ""}' for k, v in list(all_occurrences.items())[:5]]
                        results['text_content'] = {
                            'status': 'pass',
                            'message': f'All required text found in PDF ({len(phrases)} phrase(s))',
                            'details': {
                                'Phrases checked': ', '.join(phrases[:5]) + ('...' if len(phrases) > 5 else ''),
                                'Occurrences': '; '.join(occ_lines) if occ_lines else 'N/A'
                            }
                        }
                    else:
                        results['text_content'] = {
                            'status': 'fail',
                            'message': f'Some required text not found: {", ".join(missing_phrases[:3])}{"..." if len(missing_phrases) > 3 else ""}',
                            'details': {
                                'Found': ', '.join(found_phrases) if found_phrases else 'None',
                                'Missing': ', '.join(missing_phrases)
                            }
                        }
        
        doc.close()
        
    except Exception as e:
        # Return error for all checkpoints
        for checkpoint_name in checkpoints.keys():
            if checkpoints[checkpoint_name] and checkpoint_name not in results:
                results[checkpoint_name] = {
                    'status': 'error',
                    'message': f'Error during validation: {str(e)}',
                    'details': {}
                }
    
    return results

# Main UI
st.markdown('<h1 class="main-header">📄 PDF Comparison Tool</h1>', unsafe_allow_html=True)

# Mode Toggle
mode = st.radio(
    "Select Mode:",
    ["📊 Comparison Mode", "✅ Validation Mode"],
    horizontal=True,
    key="app_mode"
)

if mode == "✅ Validation Mode":
    # Validation Mode
    st.markdown("### ✅ PDF Validation & Compliance Check")
    st.markdown("Upload a PDF and select checkpoints to validate against.")
    
    # Upload PDF
    pdf_file = st.file_uploader(
        "Upload PDF to validate",
        type=['pdf'],
        key='validation_pdf',
        help="Upload the PDF document you want to validate."
    )
    
    if pdf_file:
        # Save uploaded PDF temporarily
        import tempfile
        import shutil
        from pathlib import Path
        
        temp_save_dir = Path(tempfile.gettempdir()) / "pdf_validation"
        temp_save_dir.mkdir(exist_ok=True)
        
        pdf_path = temp_save_dir / pdf_file.name
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        st.success(f"✅ PDF uploaded: {pdf_file.name}")
        
        # Checkpoints
        st.markdown("### 📋 Select Checkpoints to Validate")
        
        # Initialize session state for checkpoints if not exists
        if 'checkpoints' not in st.session_state:
            st.session_state.checkpoints = {
                'font_arial': False,
                'logo_check': False,
                'color_check': False,
                'page_count': False,
                'text_content': False
            }
        
        # Checkpoint selection
        col1, col2 = st.columns(2)
        
        with col1:
            font_check = st.checkbox(
                "🔤 Font Check: Verify required font is used (e.g. Calibri, Arial, Brandon Grotesque)",
                value=st.session_state.checkpoints.get('font_arial', False),
                key="check_font"
            )
            st.session_state.checkpoints['font_arial'] = font_check
            
            logo_check = st.checkbox(
                "🖼️ Logo Check: Verify specific logo/image exists",
                value=st.session_state.checkpoints.get('logo_check', False),
                key="check_logo"
            )
            st.session_state.checkpoints['logo_check'] = logo_check
            
            color_check = st.checkbox(
                "🎨 Color Check: Verify specific hex color codes",
                value=st.session_state.checkpoints.get('color_check', False),
                key="check_color"
            )
            st.session_state.checkpoints['color_check'] = color_check
        
        with col2:
            page_count_check = st.checkbox(
                "📄 Page Count Check: Verify page count",
                value=st.session_state.checkpoints.get('page_count', False),
                key="check_page_count"
            )
            st.session_state.checkpoints['page_count'] = page_count_check
            
            text_content_check = st.checkbox(
                "📝 Text Content Check: Verify specific text exists",
                value=st.session_state.checkpoints.get('text_content', False),
                key="check_text"
            )
            st.session_state.checkpoints['text_content'] = text_content_check
        
        # Configuration for selected checkpoints
        st.markdown("---")
        st.markdown("### ⚙️ Checkpoint Configuration")
        
        checkpoint_config = {}
        
        if font_check:
            with st.expander("🔤 Font Check Configuration"):
                required_font = st.text_input(
                    "Required font name (case-insensitive):",
                    value="Calibri",
                    key="font_name",
                    placeholder="e.g. Calibri, Arial, Brandon Grotesque"
                )
                checkpoint_config['font'] = required_font.strip()
        
        if logo_check:
            with st.expander("🖼️ Logo Check Configuration"):
                logo_file = st.file_uploader(
                    "Upload reference logo/image:",
                    type=['png', 'jpg', 'jpeg', 'gif'],
                    key="logo_reference"
                )
                if logo_file:
                    # Save logo temporarily
                    logo_path = temp_save_dir / f"logo_{logo_file.name}"
                    with open(logo_path, "wb") as f:
                        f.write(logo_file.getbuffer())
                    checkpoint_config['logo_path'] = str(logo_path)
                    st.success(f"✅ Reference logo uploaded: {logo_file.name}")
                else:
                    st.info("Upload a reference logo/image to compare against.")
        
        if color_check:
            with st.expander("🎨 Color Check Configuration"):
                color_input = st.text_input(
                    "Required hex color codes (comma-separated):",
                    value="#000000, #FFFFFF",
                    key="color_hex",
                    help="Enter hex color codes separated by commas (e.g., #FF0000, #00FF00)"
                )
                if color_input:
                    colors = [c.strip().upper() for c in color_input.split(',') if c.strip()]
                    checkpoint_config['colors'] = colors
                    # Display color previews
                    if colors:
                        cols = st.columns(len(colors))
                        for idx, color in enumerate(colors):
                            with cols[idx]:
                                st.markdown(f"**{color}**")
                                st.markdown(
                                    f'<div style="width: 50px; height: 50px; background-color: {color}; border: 1px solid #ccc;"></div>',
                                    unsafe_allow_html=True
                                )
        
        if page_count_check:
            with st.expander("📄 Page Count Check Configuration"):
                expected_pages = st.number_input(
                    "Expected page count:",
                    min_value=1,
                    value=1,
                    key="expected_pages"
                )
                checkpoint_config['page_count'] = int(expected_pages)
        
        if text_content_check:
            with st.expander("📝 Text Content Check Configuration"):
                required_text = st.text_area(
                    "Required text content (case-insensitive):",
                    key="required_text",
                    help="Enter one or more phrases that must be present in the PDF. Separate multiple phrases with commas (e.g. 'Company Name, Welcome, Section 1')."
                )
                if required_text:
                    checkpoint_config['text'] = required_text.strip()
        
        # Run validation
        if st.button("🔍 Run Validation", type="primary"):
            if not any(st.session_state.checkpoints.values()):
                st.warning("⚠️ Please select at least one checkpoint to validate.")
            else:
                # Initialize comparator if not already done
                if st.session_state.comparator is None:
                    with st.spinner("🔄 Initializing PDF comparator..."):
                        try:
                            from compare_pdfs import PDFComparator
                            st.session_state.comparator = PDFComparator()
                        except Exception as e:
                            st.error(f"❌ Failed to initialize comparator: {str(e)}")
                            st.stop()
                
                with st.spinner("🔄 Running validation checks..."):
                    results = validate_pdf_checkpoints(pdf_path, st.session_state.checkpoints, checkpoint_config)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Validation Results")
                    
                    # Summary
                    passed = sum(1 for r in results.values() if r.get('status') == 'pass')
                    total = len([k for k in st.session_state.checkpoints.keys() if st.session_state.checkpoints[k]])
                    
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("Total Checks", total)
                    with col_sum2:
                        st.metric("Passed", passed, delta=f"{passed}/{total}")
                    with col_sum3:
                        failed = total - passed
                        st.metric("Failed", failed, delta=f"{failed}/{total}", delta_color="inverse")
                    
                    # Detailed results
                    st.markdown("#### Detailed Results:")
                    
                    for checkpoint_name, result in results.items():
                        if not st.session_state.checkpoints.get(checkpoint_name, False):
                            continue
                        
                        status = result.get('status', 'unknown')
                        message = result.get('message', '')
                        details = result.get('details', {})
                        
                        if status == 'pass':
                            st.success(f"✅ **{checkpoint_name.replace('_', ' ').title()}**: {message}")
                        elif status == 'fail':
                            st.error(f"❌ **{checkpoint_name.replace('_', ' ').title()}**: {message}")
                        else:
                            st.warning(f"⚠️ **{checkpoint_name.replace('_', ' ').title()}**: {message}")
                        
                        if details:
                            with st.expander(f"View details for {checkpoint_name.replace('_', ' ').title()}"):
                                for key, value in details.items():
                                    st.write(f"**{key}**: {value}")
                    
                    # Overall status
                    st.markdown("---")
                    if passed == total:
                        st.success(f"🎉 **All checks passed!** ({passed}/{total})")
                    elif passed > 0:
                        st.warning(f"⚠️ **Partial pass:** {passed}/{total} checks passed")
                    else:
                        st.error(f"❌ **All checks failed!** (0/{total})")

else:
    # Comparison Mode (existing code)
    st.markdown("### Upload PDF files or paste HTML to compare and see differences")
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📑 Document 1 (Original)")
        pdf1_file = st.file_uploader(
            "Upload first PDF",
            type=['pdf'],
            key='pdf1',
            help="Upload the original or first PDF document"
        )
    
    with col2:
        st.subheader("📑 Document 2 (To Compare)")
        
        # Option to choose between PDF upload or HTML paste
        doc2_input_type = st.radio(
            "Input type:",
            ["Upload PDF", "Paste HTML"],
            key="doc2_input_type",
            horizontal=True
        )
        
        pdf2_file = None
        html_content = None
        
        if doc2_input_type == "Upload PDF":
            pdf2_file = st.file_uploader(
                "Upload second PDF",
                type=['pdf'],
                key='pdf2',
                help="Upload the second PDF document. Differences will be highlighted in this PDF."
            )
        else:
            st.markdown("**Paste HTML content:**")
            st.info("ℹ️ **Note**: Full HTML and CSS support via headless browser. First-time use will install browser automatically (~100MB). %%[IF (@var == \"value\") THEN]%% blocks are resolved by variable state: only the block matching the chosen state (e.g. @mfsAppDownloaded = false) is shown.")
            html_content = st.text_area(
                "HTML Content",
                key='html_content',
                height=300,
                help="Paste your HTML here. It will be converted to PDF for comparison. AMPscript %%[IF]%% blocks with the same variable are collapsed to one box (default: state \"false\").",
                placeholder="""<html>
<head>
    <title>Document</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #333; }
        p { line-height: 1.6; }
    </style>
</head>
<body>
    <h1>Your Content Here</h1>
    <p>Paste your HTML content...</p>
</body>
</html>"""
            )

            if html_content and html_content.strip():
                # --- Litmus Tracking Check ---
                has_litmus = check_litmus_tracking(html_content)
                litmus_msg = "✅ Litmus tracking code found" if has_litmus else "❌ Litmus tracking code missing"
                if has_litmus:
                    st.success(litmus_msg)
                else:
                    st.error(litmus_msg)
                # -----------------------------

                # --- Link/Image Title Attribute Check ---
                title_errors = check_missing_title_attributes(html_content)
                if not title_errors:
                     st.success("✅ all Links and Images have Title attributes.")
                else:
                    with st.expander(f"⚠️ Found {len(title_errors)} Missing Title Attributes", expanded=False):
                        for err in title_errors:
                            st.write(err)
                # ----------------------------------------
                
                # --- Image Alt vs Link Alias Check ---
                alt_alias_errors = check_image_alt_matches_link_alias(html_content)
                if not alt_alias_errors:
                     st.success("✅ All images inside links match their link aliases.")
                else:
                    with st.expander(f"⚠️ Found {len(alt_alias_errors)} Link Alias/Alt Text Mismatches", expanded=False):
                        for err in alt_alias_errors:
                            st.write(err)
                # -------------------------------------

                # Detect variables in AMPscript and let user choose state per variable
                ampscript_vars = get_ampscript_variables(html_content)
                chosen_state: Optional[Dict[str, str]] = None
                if ampscript_vars:
                    if "ampscript_chosen_state" not in st.session_state:
                        st.session_state.ampscript_chosen_state = {}
                    # Keep only variables present in current HTML
                    st.session_state.ampscript_chosen_state = {
                        k: v for k, v in st.session_state.ampscript_chosen_state.items()
                        if k in ampscript_vars
                    }
                    st.markdown("**Variable states** (choose which branch to show per variable)")
                    cols = st.columns(min(len(ampscript_vars), 3))
                    for idx, (var_name, states) in enumerate(ampscript_vars.items()):
                        with cols[idx % len(cols)]:
                            default = next((s for s in states if s.lower() == "false"), states[0])
                            current = st.session_state.ampscript_chosen_state.get(var_name, default)
                            if current not in states:
                                current = default
                            choice = st.selectbox(
                                f"**@{var_name}**",
                                options=states,
                                index=states.index(current) if current in states else 0,
                                key=f"ampscript_var_{var_name}",
                            )
                            st.session_state.ampscript_chosen_state[var_name] = choice
                    chosen_state = dict(st.session_state.ampscript_chosen_state)
                else:
                    chosen_state = None

                st.markdown("**HTML Preview:** (only blocks matching selected variable state are shown)")
                try:
                    resolved_preview = resolve_and_strip_ampscript(html_content, chosen_state=chosen_state)
                    st.components.v1.html(resolved_preview, height=400, scrolling=True)
                except ValueError as e:
                    st.warning(f"⚠️ Unbalanced AMPscript: {e} Check %%[IF]%%/%%[ELSE]%%/%%[ENDIF]%% tags.")
                    st.components.v1.html(html_content, height=400, scrolling=True)
                except Exception as e:
                    st.info("HTML preview not available. The HTML will still be converted to PDF for comparison.")
    
    # Compare button
    if st.button("🔍 Compare Documents", type="primary"):
        # Validate inputs
        if pdf1_file is None:
            st.error("⚠️ Please upload Document 1 (PDF).")
        elif doc2_input_type == "Upload PDF" and pdf2_file is None:
            st.error("⚠️ Please upload Document 2 (PDF) or switch to HTML input.")
        elif doc2_input_type == "Paste HTML" and (not html_content or not html_content.strip()):
            st.error("⚠️ Please paste HTML content for Document 2.")
        else:
            with st.spinner("Processing documents... This may take a moment."):
                try:
                    # Create temporary directory for uploaded files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save Document 1
                        pdf1_path = save_uploaded_file(pdf1_file, temp_dir)
                        
                        # Initialize comparator early (needed for extraction below)
                        if st.session_state.comparator is None:
                            print("[DEBUG] Initializing PDFComparator")
                            st.session_state.comparator = PDFComparator()
                        
                        # Handle Document 2 - either PDF or HTML
                        if doc2_input_type == "Upload PDF":
                            print(f"[DEBUG] Processing Doc2 as PDF: {pdf2_file.name}")
                            pdf2_path = save_uploaded_file(pdf2_file, temp_dir)
                            pdf2_name = pdf2_file.name
                            text2 = st.session_state.comparator.extract_text_from_pdf(pdf2_path)
                            print(f"[DEBUG] Extracted {len(text2)} chars from PDF2")
                            is_doc2_html = False
                            resolved_html_for_diff = None
                        else:
                            # Resolve AMPscript first so we compare against the SAME branch the user sees
                            print("[DEBUG] Processing Doc2 as Resolved HTML Preview")
                            chosen_state = st.session_state.get("ampscript_chosen_state")
                            resolved_html_for_diff = resolve_and_strip_ampscript(html_content, chosen_state=chosen_state)
                            
                            # Direct text extraction from RESOLVED HTML
                            text2 = st.session_state.comparator.extract_text_from_html(resolved_html_for_diff)
                            print(f"[DEBUG] Extracted {len(text2)} chars from Resolved HTML")
                            is_doc2_html = True
                            pdf2_name = "HTML Document"
                            
                            # Convert resolved HTML into a PDF for visual reference in Side-by-Side view
                            with st.spinner("Preparing HTML for preview..."):
                                print("[DEBUG] Converting Resolved HTML to PDF for visual preview")
                                pdf2_path = os.path.join(temp_dir, "html_converted.pdf")
                                
                                # Match PDF1 size
                                try:
                                    import fitz
                                    doc1 = fitz.open(pdf1_path)
                                    pdf1_width_inches = doc1[0].rect.width / 72.0
                                    pdf1_height_inches = doc1[0].rect.height / 72.0
                                    doc1.close()
                                except Exception as e:
                                    print(f"[DEBUG] PDF1 size extraction failed: {e}")
                                    pdf1_width_inches, pdf1_height_inches = 8.5, 11.0
                                    
                                if html_to_pdf(resolved_html_for_diff, pdf2_path, chosen_state=None,
                                           page_width=pdf1_width_inches, page_height=pdf1_height_inches):
                                    print("[DEBUG] HTML to PDF conversion successful")
                                else:
                                    print("[DEBUG] HTML to PDF conversion failed")

                        # Ensure comparator is initialized (already done above, but for safety)
                        if st.session_state.comparator is None:
                            print("[DEBUG] Initializing PDFComparator (safety fallback)")
                            st.session_state.comparator = PDFComparator()
                        
                        # Extract text from PDF1
                        print("[DEBUG] Extracting text from PDF1")
                        text1 = st.session_state.comparator.extract_text_from_pdf(pdf1_path)
                        print(f"[DEBUG] Extracted {len(text1)} chars from PDF1")
                        
                        # Compare semantically
                        st.info("🔄 Performing semantic comparison...")
                        print("[DEBUG] Starting Semantic Comparison")
                        text_diff = st.session_state.comparator.find_text_differences_chunk_based(text1, text2)
                        print(f"[DEBUG] Comparison complete: {text_diff.get('added_lines')} added, {text_diff.get('removed_lines')} removed")
                        
                        semantic_sim_max, semantic_sim_avg = st.session_state.comparator.calculate_semantic_similarity(text1, text2)
                        print(f"[DEBUG] Semantic Similarity: {semantic_sim_avg:.2f}")
                        
                        # Highlighting Phase (The "Where")
                        if is_doc2_html:
                            print("[DEBUG] Highlighting HTML content (Additions Only)")
                            # In this split-view model, HTML ONLY shows additions
                            from html_processor import highlight_html_content
                            highlighted_html = highlight_html_content(
                                resolved_html_for_diff, 
                                text_diff.get('added_chunks', [])
                            )
                            st.session_state.highlighted_html = highlighted_html
                            print("[DEBUG] HTML highlighting complete")
                        else:
                            st.session_state.highlighted_html = None
                        
                        # Extract and compare images
                        image_comparison = None
                        images1 = None
                        images2 = None
                        try:
                            images1 = st.session_state.comparator.extract_images_from_pdf(pdf1_path)
                            images2 = st.session_state.comparator.extract_images_from_pdf(pdf2_path)
                            if images1 or images2:
                                image_comparison = st.session_state.comparator.compare_images(images1, images2)
                        except Exception as e:
                            st.warning(f"⚠️ Image comparison skipped: {str(e)}")
                        
                        # Extract and compare fonts
                        font_comparison = None
                        try:
                            fonts1 = st.session_state.comparator.extract_fonts_from_pdf(pdf1_path)
                            fonts2 = st.session_state.comparator.extract_fonts_from_pdf(pdf2_path)
                            
                            # Debug: Show font extraction results
                            if fonts1.get('unique_count', 0) == 0 and fonts2.get('unique_count', 0) == 0:
                                st.warning("⚠️ No fonts detected in either PDF. This might indicate:")
                                st.warning("  - PDFs contain only images/scanned content")
                                st.warning("  - Fonts are embedded in a non-standard format")
                                st.warning("  - PDFs are password-protected or corrupted")
                            else:
                                st.info(f"✓ Detected {fonts1.get('unique_count', 0)} unique fonts in Document 1, {fonts2.get('unique_count', 0)} in Document 2")
                            
                            font_comparison = st.session_state.comparator.compare_fonts(fonts1, fonts2)
                        except Exception as e:
                            st.error(f"❌ Font comparison failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language='python')
                        
                        # Generate report
                        report = st.session_state.comparator.generate_comparison_report(
                            text1,
                            text2,
                            pdf1_file.name,
                            pdf2_name,
                            semantic_sim_max,
                            semantic_sim_avg,
                            text_diff,
                            image_comparison,
                            font_comparison
                        )
                        
                        # Store results in session state
                        # Note: We need to save PDFs to a persistent location for highlighting
                        temp_save_dir = Path(tempfile.gettempdir()) / "pdf_comparison"
                        temp_save_dir.mkdir(exist_ok=True)
                        
                        saved_pdf1 = temp_save_dir / f"pdf1_{pdf1_file.name}"
                        saved_pdf2_name = f"pdf2_{pdf2_name.replace(' ', '_')}.pdf"
                        saved_pdf2 = temp_save_dir / saved_pdf2_name
                        shutil.copy(pdf1_path, saved_pdf1)
                        shutil.copy(pdf2_path, saved_pdf2)
                        
                        st.session_state.report = report
                        st.session_state.text_diff = text_diff
                        st.session_state.pdf1_path = str(saved_pdf1)
                        st.session_state.pdf2_path = str(saved_pdf2)
                        st.session_state.image_comparison = image_comparison
                        st.session_state.font_comparison = font_comparison
                        st.session_state.images1 = images1
                        st.session_state.images2 = images2
                        st.session_state.is_doc2_html = is_doc2_html # NEW: Store for UI rendering
                        
                        # Render PDF pages for visual comparison
                        try:
                            with st.spinner("Rendering PDF pages for visual comparison..."):
                                pdf_pages1 = st.session_state.comparator.render_all_pdf_pages(pdf1_path, max_pages=20)
                                pdf_pages2 = st.session_state.comparator.render_all_pdf_pages(pdf2_path, max_pages=20)
                                st.session_state.pdf_pages1 = pdf_pages1
                                st.session_state.pdf_pages2 = pdf_pages2
                        except Exception as e:
                            st.warning(f"⚠️ PDF rendering skipped: {str(e)}")
                            st.session_state.pdf_pages1 = None
                            st.session_state.pdf_pages2 = None
                        
                        # Invalidate highlighted cache so side-by-side view uses new Doc 1/Doc 2
                        if 'highlighted_pdf1' in st.session_state:
                            st.session_state.highlighted_pdf1 = None
                        if 'highlighted_pages1' in st.session_state:
                            st.session_state.highlighted_pages1 = None
                        if 'highlighted_pages2' in st.session_state:
                            st.session_state.highlighted_pages2 = None
                        for key in ('highlighted_pdf1_path', 'highlighted_pdf2_path'):
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.session_state.comparison_done = True
                        
                        st.success("✅ Comparison complete!")
                except Exception as e:
                    st.error(f"❌ Error during comparison: {str(e)}")
                    st.session_state.comparison_done = False
    
    # Display results - Consolidated View
    if st.session_state.comparison_done and st.session_state.report:
        st.markdown("---")
        st.markdown("## 📊 Comparison Results - Side by Side View")
        
        # Generate highlighted PDFs if not already done
        if 'highlighted_pdf1' not in st.session_state or st.session_state.highlighted_pdf1 is None:
            with st.spinner("🔄 Generating highlighted PDFs with differences..."):
                try:
                    # Create highlighted version of PDF1 (shows removals)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        highlighted_pdf1_path = tmp_file.name
                        highlight_pdf_differences(
                            st.session_state.pdf1_path,
                            text_diff, # Processes both additions/removals; we use it here to show removals in PDF1
                            highlighted_pdf1_path
                        )
                        st.session_state.highlighted_pdf1_path = highlighted_pdf1_path
                    
                    # Create highlighted version of PDF2 ONLY if it's not HTML 
                    if not is_doc2_html:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            highlighted_pdf2_path = tmp_file.name
                            highlight_pdf_differences(
                                st.session_state.pdf2_path,
                                text_diff, # Shows additions/changes
                                highlighted_pdf2_path
                            )
                            st.session_state.highlighted_pdf2_path = highlighted_pdf2_path
                    else:
                        highlighted_pdf2_path = st.session_state.pdf2_path
                        st.session_state.highlighted_pdf2_path = st.session_state.pdf2_path
                    
                    # Render highlighted PDF pages
                    highlighted_pages1 = st.session_state.comparator.render_all_pdf_pages(
                        highlighted_pdf1_path, max_pages=20
                    )
                    highlighted_pages2 = st.session_state.comparator.render_all_pdf_pages(
                        highlighted_pdf2_path, max_pages=20
                    )
                    st.session_state.highlighted_pages1 = highlighted_pages1
                    st.session_state.highlighted_pages2 = highlighted_pages2
                    st.session_state.highlighted_pdf1 = True
                except Exception as e:
                    st.warning(f"⚠️ Could not generate highlighted PDFs: {str(e)}")
                    # Fallback to regular pages
                    st.session_state.highlighted_pages1 = st.session_state.pdf_pages1
                    st.session_state.highlighted_pages2 = st.session_state.pdf_pages2
        
        # Summary Metrics
        st.markdown("### 📈 Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Calculate semantic similarity (using same normalization as comparison)
            try:
                text1 = st.session_state.comparator.extract_text_from_pdf(st.session_state.pdf1_path)
                text2 = st.session_state.comparator.extract_text_from_pdf(st.session_state.pdf2_path)
                text1 = normalize_for_comparison(text1)
                text2 = normalize_for_comparison(text2)
                _, semantic_sim = st.session_state.comparator.calculate_semantic_similarity(text1, text2)
                semantic_sim_pct = semantic_sim * 100
            except:
                semantic_sim_pct = 0
            st.metric("Similarity", f"{semantic_sim_pct:.1f}%")
        
        with col2:
            added = st.session_state.text_diff.get('added_lines', 0) if st.session_state.text_diff else 0
            st.metric("Added Lines", added)
        
        with col3:
            removed = st.session_state.text_diff.get('removed_lines', 0) if st.session_state.text_diff else 0
            st.metric("Removed Lines", removed)
        
        with col4:
            changed = st.session_state.text_diff.get('changed_lines', 0) if st.session_state.text_diff else 0
            st.metric("Changed Lines", changed)
        
        with col5:
            if st.session_state.image_comparison:
                img_sim = st.session_state.image_comparison.get('similarity_score', 0) * 100
                st.metric("Image Similarity", f"{img_sim:.1f}%")
            else:
                st.metric("Image Similarity", "N/A")
        
        # PDF vs HTML: show presence summary (line diff is 0 by design)
        td = st.session_state.text_diff or {}
        if td.get("comparison_mode") == "pdf_vs_html":
            total = td.get("html_blocks_total", 0)
            in_pdf = td.get("html_blocks_in_pdf_count", 0)
            not_in = td.get("html_blocks_not_in_pdf", [])
            st.info(f"**PDF vs HTML comparison:** {in_pdf} of {total} HTML content blocks found in PDF." + (" No line diff (presence-based)." if total else ""))
            if not_in:
                with st.expander("HTML blocks not found in PDF"):
                    for i, block in enumerate(not_in[:50], 1):
                        st.text(block[:200] + ("..." if len(block) > 200 else ""))
                    if len(not_in) > 50:
                        st.caption(f"... and {len(not_in) - 50} more.")
        
        st.markdown("---")
        
        # Side-by-side PDF comparison with highlights
        st.markdown("### 📄 Side-by-Side PDF Comparison")
        st.markdown("**Legend:** 🔴 Red = Removed (PDF 1) | 🟡 Yellow = Added (PDF 2) | 🟠 Orange = Changed (PDF 2)")
        
        # Use highlighted pages if available, otherwise use regular pages
        pages1 = st.session_state.get('highlighted_pages1', st.session_state.pdf_pages1)
        pages2 = st.session_state.get('highlighted_pages2', st.session_state.pdf_pages2)
        
        if pages1 and pages2:
            # Page selector
            max_pages = max(len(pages1), len(pages2))
            if max_pages > 0:
                # Navigation buttons
                col_prev, col_info, col_next, col_download = st.columns([1, 2, 1, 1])
                with col_prev:
                    if st.session_state.current_page_view > 1:
                        if st.button("◀ Previous", key="prev_page"):
                            st.session_state.current_page_view -= 1
                            st.rerun()
                
                with col_info:
                    page_to_view = st.selectbox(
                        "Select page to compare:",
                        range(1, max_pages + 1),
                        index=st.session_state.current_page_view - 1,
                        key="pdf_page_selector",
                        on_change=lambda: setattr(st.session_state, 'current_page_view', st.session_state.pdf_page_selector)
                    )
                    st.session_state.current_page_view = page_to_view
                
                with col_next:
                    if st.session_state.current_page_view < max_pages:
                        if st.button("Next ▶", key="next_page"):
                            st.session_state.current_page_view += 1
                            st.rerun()
                
                with col_download:
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        if st.session_state.get('highlighted_pdf1_path'):
                            try:
                                with open(st.session_state.highlighted_pdf1_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    label="📥 PDF 1 (Removals)",
                                    data=pdf_bytes,
                                    file_name="pdf1_highlighted.pdf",
                                    mime="application/pdf",
                                    key="download_highlighted_pdf1"
                                )
                            except:
                                pass
                    with col_dl2:
                        if st.session_state.get('is_doc2_html') and st.session_state.get('highlighted_html'):
                            # Generate High-Precision PDF from Highlighted HTML
                            with st.spinner("Generating High-Precision PDF..."):
                                try:
                                    # Use stored dimensions if available
                                    pw, ph = 8.5, 11.0
                                    try:
                                        import fitz
                                        doc1 = fitz.open(st.session_state.pdf1_path)
                                        pw = doc1[0].rect.width / 72.0
                                        ph = doc1[0].rect.height / 72.0
                                        doc1.close()
                                    except: pass
                                    
                                    pdf_bytes = export_highlighted_html_to_pdf(st.session_state.highlighted_html, pw, ph)
                                    st.download_button(
                                        label="📥 PDF 2 (High-Precision)",
                                        data=pdf_bytes,
                                        file_name="doc2_semantic_highlights.pdf",
                                        mime="application/pdf",
                                        key="download_highlighted_html_pdf"
                                    )
                                except Exception as e:
                                    st.error(f"Failed to export PDF: {e}")
                        elif st.session_state.get('highlighted_pdf2_path'):
                            try:
                                with open(st.session_state.highlighted_pdf2_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    label="📥 PDF 2 (Additions)",
                                    data=pdf_bytes,
                                    file_name="pdf2_highlighted.pdf",
                                    mime="application/pdf",
                                    key="download_highlighted_pdf2"
                                )
                            except:
                                pass
                
                # Get page images
                page1_img = None
                page2_img = None
                
                for page_data in pages1:
                    if page_data['page_num'] == page_to_view:
                        page1_img = page_data['image']
                        break
                
                for page_data in pages2:
                    if page_data['page_num'] == page_to_view:
                        page2_img = page_data['image']
                        break
                
                if page1_img or page2_img:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### 📄 Document 1 - Page {page_to_view} (removals highlighted)")
                        if page1_img:
                            st.image(page1_img, width='stretch', caption=f"Page {page_to_view} from Document 1 - Removed content highlighted in red")
                        else:
                            st.info("Page not available in Document 1")
                    
                    with col2:
                        if st.session_state.get('is_doc2_html') and st.session_state.get('highlighted_html'):
                            st.markdown(f"### 🌐 Document 2 - HTML Source (with semantic highlights)")
                            st.markdown("*Additions and changes found by comparing semantic content blocks.*")
                            st.components.v1.html(
                                f"""
                                <style>
                                    body {{ font-family: sans-serif; padding: 20px; }}
                                    .diff-added {{ background-color: #d4edda; border-bottom: 2px solid #28a745; padding: 2px; border-radius: 2px; }}
                                </style>
                                {st.session_state.highlighted_html}
                                """,
                                height=2000,
                                scrolling=True
                            )
                        else:
                            st.markdown(f"### 📄 Document 2 - Page {page_to_view} (with highlights)")
                            if page2_img:
                                st.image(page2_img, width='stretch', caption=f"Page {page_to_view} from Document 2 - Differences highlighted")
                            else:
                                st.info("Page not available in Document 2")
                else:
                    st.warning(f"Page {page_to_view} not found in either document.")
            else:
                st.info("No pages available for comparison.")
        else:
            st.info("PDF pages not rendered. Please run the comparison again.")
        
        # Collapsible detailed statistics
        with st.expander("📊 Detailed Statistics & Analysis"):
            # Text differences summary
            if st.session_state.text_diff:
                st.markdown("#### Text Differences")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Added Lines", st.session_state.text_diff.get('added_lines', 0))
                with col2:
                    st.metric("Removed Lines", st.session_state.text_diff.get('removed_lines', 0))
                with col3:
                    st.metric("Changed Lines", st.session_state.text_diff.get('changed_lines', 0))
            
            # Image comparison summary
            if st.session_state.image_comparison:
                st.markdown("#### Image Comparison")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images (Doc1)", st.session_state.image_comparison.get('total_images_1', 0))
                with col2:
                    st.metric("Total Images (Doc2)", st.session_state.image_comparison.get('total_images_2', 0))
                with col3:
                    st.metric("Similar Images", len(st.session_state.image_comparison.get('similar_images', [])))
                with col4:
                    spacing_stats = st.session_state.image_comparison.get('spacing_stats', {})
                    same_spacing = spacing_stats.get('same_spacing', 0)
                    total_compared = spacing_stats.get('total_compared', 0)
                    if total_compared > 0:
                        spacing_pct = (same_spacing / total_compared) * 100
                        st.metric("Same Spacing", f"{same_spacing}/{total_compared} ({spacing_pct:.0f}%)")
            
            # Font comparison summary
            if st.session_state.font_comparison:
                st.markdown("#### Font Comparison")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fonts (Doc1)", st.session_state.font_comparison.get('font_count_1', 0))
                with col2:
                    st.metric("Fonts (Doc2)", st.session_state.font_comparison.get('font_count_2', 0))
                with col3:
                    st.metric("Common Fonts", st.session_state.font_comparison.get('common_count', 0))
            
            # Download full report
            st.markdown("---")
            st.download_button(
                label="📥 Download Full Text Report",
                data=st.session_state.report,
                file_name="comparison_report.txt",
                mime="text/plain",
                key="download_report"
            )

# Footer (shared for both modes)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Hugging Face Models • No API keys required"
    "</div>",
    unsafe_allow_html=True
)
