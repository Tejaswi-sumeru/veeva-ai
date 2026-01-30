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
from compare_pdfs import PDFComparator
import io

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

def html_to_pdf(html_content: str, output_path: str) -> bool:
    """
    Convert HTML content to PDF using Playwright (headless browser).
    Playwright handles browser installation automatically.
    Strips any content between %%[ and ]%% before conversion.
    
    Args:
        html_content: HTML string to convert
        output_path: Path where PDF will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from playwright.sync_api import sync_playwright
        import re
        
        html_processed = html_content.strip()
        segment_pat = re.compile(r"%%\[(.*?)\]%%", re.DOTALL)

        def is_if_open(inner):
            s = (inner or "").strip().upper()
            return s.startswith("IF")

        def is_endif(inner):
            s = (inner or "").strip().upper()
            return s == "ENDIF" or "ENDIF" in s

        while True:
            segments = list(segment_pat.finditer(html_processed))
            block_starts = []
            i = 0
            while i < len(segments):
                m = segments[i]
                if is_if_open(m.group(1)):
                    start_pos = m.start()
                    block_content_start = m.end()
                    i += 1
                    while i < len(segments) and not is_endif(segments[i].group(1)):
                        i += 1
                    if i < len(segments):
                        end_pos = segments[i].end()
                        block_starts.append((start_pos, end_pos))
                        i += 1
                    continue
                i += 1
            removed = False
            for j in range(len(block_starts) - 1):
                _, end1 = block_starts[j]
                start2, end2 = block_starts[j + 1]
                between = html_processed[end1:start2]
                if between.strip() == "":
                    html_processed = html_processed[:start2] + html_processed[end2:]
                    removed = True
                    break
            if not removed:
                break

        html_processed = re.sub(r"%%\[.*?\]%%", "", html_processed, flags=re.DOTALL)
        html_processed = html_processed.strip()

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
                page.pdf(
                    path=output_path,
                    width=f"{inch_w}in",
                    height=f"{inch_h}in",
                    print_background=True,
                    margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"}
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
    Only highlights specific removed instances, not all occurrences.
    
    Args:
        pdf1_path: Path to original PDF1
        text_diff: Dictionary with line differences
        output_path: Path to save highlighted PDF
    """
    try:
        import fitz  # PyMuPDF
        import difflib
    except ImportError:
        st.error("PyMuPDF (fitz) is required for PDF highlighting. Install with: pip install PyMuPDF")
        return False
    
    try:
        # Open PDF1
        doc = fitz.open(pdf1_path)
        
        # Get removed lines from text_diff
        line_differences = text_diff.get('line_differences', [])
        removed_lines = [d for d in line_differences if d['type'] == 'removed']
        
        highlighted_count = 0
        used_highlights = set()  # Track which text we've already highlighted
        
        def highlight_unique_text(text_to_find, color, context_before="", context_after="", max_length=50):
            """
            Highlight text only once, using context to make it unique.
            This ensures we only highlight the specific instance, not all occurrences.
            """
            nonlocal highlighted_count
            
            # Normalize text for comparison
            text_normalized = text_to_find.strip().lower()
            if len(text_normalized) < 2:
                return False
            
            # Create a unique key for this highlight (include context)
            highlight_key = f"{context_before.lower()}|{text_normalized}|{context_after.lower()}"
            if highlight_key in used_highlights:
                return False  # Already highlighted this specific instance
            
            found_any = False
            
            # Try searching with context first (more precise)
            if context_before or context_after:
                # Build search pattern with context
                search_patterns = []
                if context_before and context_after:
                    # Try full context
                    search_patterns.append(f"{context_before} {text_to_find} {context_after}")
                if context_before:
                    search_patterns.append(f"{context_before} {text_to_find}")
                if context_after:
                    search_patterns.append(f"{text_to_find} {context_after}")
                
                for pattern in search_patterns:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        # Search for the pattern
                        pattern_instances = page.search_for(pattern.strip(), flags=fitz.TEXT_DEHYPHENATE)
                        
                        if pattern_instances:
                            # Found with context - now find just the target text near this location
                            for pattern_inst in pattern_instances[:1]:  # Only first match
                                # Search for target text on same page
                                target_instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE)
                                
                                for target_inst in target_instances:
                                    # Check if target is near the pattern match (same area)
                                    y_distance = abs(target_inst.y0 - pattern_inst.y0)
                                    x_distance = abs(target_inst.x0 - pattern_inst.x0)
                                    
                                    # If target is within reasonable distance of pattern
                                    if y_distance < 30 and x_distance < 600:
                                        try:
                                            highlight = page.add_highlight_annot(target_inst)
                                            highlight.set_colors(stroke=color)
                                            highlight.set_opacity(0.3)
                                            highlight.update()
                                            highlighted_count += 1
                                            used_highlights.add(highlight_key)
                                            found_any = True
                                            return True  # Found and highlighted, exit
                                        except:
                                            pass
                                if found_any:
                                    break
                            if found_any:
                                break
                    if found_any:
                        break
            
            # If not found with context, try without context but only highlight once
            if not found_any:
                # Use a simpler key for tracking
                simple_key = f"{text_normalized}"
                if simple_key not in used_highlights:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text_instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE)
                        
                        if text_instances:
                            # Only highlight the FIRST instance found
                            try:
                                highlight = page.add_highlight_annot(text_instances[0])
                                highlight.set_colors(stroke=color)
                                highlight.set_opacity(0.3)
                                highlight.update()
                                highlighted_count += 1
                                used_highlights.add(simple_key)
                                used_highlights.add(highlight_key)  # Also mark the full key
                                found_any = True
                            except:
                                pass
                            break  # Only highlight once
            
            return found_any
        
        # Extract full text from PDF1 to get context
        pdf1_full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pdf1_full_text.extend(page.get_text().splitlines())
        
        # Highlight removed lines (red) - use line number to get context
        for diff in removed_lines:
            content = diff.get('content', '').strip()
            line_num = diff.get('doc1_line')
            
            if content and len(content) > 3 and line_num:
                # Get context from surrounding lines
                context_before = ""
                context_after = ""
                
                if line_num > 1 and line_num <= len(pdf1_full_text):
                    # Get previous line as context
                    prev_line = pdf1_full_text[line_num - 2] if line_num > 1 else ""
                    if prev_line.strip():
                        context_before = prev_line.strip()[:30]  # First 30 chars
                
                if line_num < len(pdf1_full_text):
                    # Get next line as context
                    next_line = pdf1_full_text[line_num] if line_num < len(pdf1_full_text) else ""
                    if next_line.strip():
                        context_after = next_line.strip()[:30]  # First 30 chars
                
                # Use a unique portion of the content
                if len(content) > 20:
                    search_text = content[:20]  # Use first 20 chars for uniqueness
                else:
                    search_text = content
                
                # Use red color for removed content
                highlight_unique_text(search_text, [1, 0, 0], context_before, context_after)
        
        # Save highlighted PDF
        doc.save(output_path)
        doc.close()
        
        if highlighted_count == 0:
            st.warning("⚠️ No text matches found for highlighting removals. The PDF text might be in images or have different formatting.")
        else:
            st.info(f"✅ Highlighted {highlighted_count} unique removal(s) in PDF 1.")
        
        return True
        
    except Exception as e:
        st.error(f"Error creating highlighted PDF 1: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def highlight_pdf_differences(pdf2_path, text_diff, output_path):
    """
    Create a highlighted version of PDF2 showing differences.
    Only highlights specific changed instances, not all occurrences.
    
    Args:
        pdf2_path: Path to original PDF2
        text_diff: Dictionary with line differences
        output_path: Path to save highlighted PDF
    """
    try:
        import fitz  # PyMuPDF
        import difflib
    except ImportError:
        st.error("PyMuPDF (fitz) is required for PDF highlighting. Install with: pip install PyMuPDF")
        return False
    
    try:
        # Open PDF2
        doc = fitz.open(pdf2_path)
        
        # Get added and changed lines from text_diff
        line_differences = text_diff.get('line_differences', [])
        added_lines = [d for d in line_differences if d['type'] == 'added']
        changed_lines = [d for d in line_differences if d['type'] == 'changed']
        
        highlighted_count = 0
        used_highlights = set()  # Track which text we've already highlighted
        
        def find_word_differences(old_text, new_text):
            """Find specific words that changed between old and new text."""
            old_words = old_text.split()
            new_words = new_text.split()
            
            # Use SequenceMatcher to find changed words
            matcher = difflib.SequenceMatcher(None, old_words, new_words)
            changed_word_groups = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace' or tag == 'insert':
                    # Words that were added or changed
                    if j2 > j1:
                        changed_word_groups.append(' '.join(new_words[j1:j2]))
            
            return changed_word_groups
        
        def highlight_unique_text(text_to_find, color, context_before="", context_after="", max_length=50):
            """
            Highlight text only once, using context to make it unique.
            This ensures we only highlight the specific instance, not all occurrences.
            """
            nonlocal highlighted_count
            
            # Normalize text for comparison
            text_normalized = text_to_find.strip().lower()
            if len(text_normalized) < 2:
                return False
            
            # Create a unique key for this highlight (include context)
            highlight_key = f"{context_before.lower()}|{text_normalized}|{context_after.lower()}"
            if highlight_key in used_highlights:
                return False  # Already highlighted this specific instance
            
            found_any = False
            
            # Try searching with context first (more precise)
            if context_before or context_after:
                # Build search pattern with context
                search_patterns = []
                if context_before and context_after:
                    # Try full context
                    search_patterns.append(f"{context_before} {text_to_find} {context_after}")
                if context_before:
                    search_patterns.append(f"{context_before} {text_to_find}")
                if context_after:
                    search_patterns.append(f"{text_to_find} {context_after}")
                
                for pattern in search_patterns:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        # Search for the pattern
                        pattern_instances = page.search_for(pattern.strip(), flags=fitz.TEXT_DEHYPHENATE)
                        
                        if pattern_instances:
                            # Found with context - now find just the target text near this location
                            for pattern_inst in pattern_instances[:1]:  # Only first match
                                # Search for target text on same page
                                target_instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE)
                                
                                for target_inst in target_instances:
                                    # Check if target is near the pattern match (same area)
                                    y_distance = abs(target_inst.y0 - pattern_inst.y0)
                                    x_distance = abs(target_inst.x0 - pattern_inst.x0)
                                    
                                    # If target is within reasonable distance of pattern
                                    if y_distance < 30 and x_distance < 600:
                                        try:
                                            highlight = page.add_highlight_annot(target_inst)
                                            highlight.set_colors(stroke=color)
                                            highlight.set_opacity(0.3)
                                            highlight.update()
                                            highlighted_count += 1
                                            used_highlights.add(highlight_key)
                                            found_any = True
                                            return True  # Found and highlighted, exit
                                        except:
                                            pass
                                if found_any:
                                    break
                            if found_any:
                                break
                    if found_any:
                        break
            
            # If not found with context, try without context but only highlight once
            if not found_any:
                # Use a simpler key for tracking
                simple_key = f"{text_normalized}"
                if simple_key not in used_highlights:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text_instances = page.search_for(text_to_find, flags=fitz.TEXT_DEHYPHENATE)
                        
                        if text_instances:
                            # Only highlight the FIRST instance found
                            try:
                                highlight = page.add_highlight_annot(text_instances[0])
                                highlight.set_colors(stroke=color)
                                highlight.set_opacity(0.3)
                                highlight.update()
                                highlighted_count += 1
                                used_highlights.add(simple_key)
                                used_highlights.add(highlight_key)  # Also mark the full key
                                found_any = True
                            except:
                                pass
                            break  # Only highlight once
            
            return found_any
        
        # Extract full text from PDF2 to get context
        pdf2_full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pdf2_full_text.extend(page.get_text().splitlines())
        
        # Highlight added lines (yellow) - use line number to get context
        for diff in added_lines:
            content = diff.get('content', '').strip()
            line_num = diff.get('doc2_line')
            
            if content and len(content) > 3 and line_num:
                # Get context from surrounding lines
                context_before = ""
                context_after = ""
                
                if line_num > 1 and line_num <= len(pdf2_full_text):
                    # Get previous line as context
                    prev_line = pdf2_full_text[line_num - 2] if line_num > 1 else ""
                    if prev_line.strip():
                        context_before = prev_line.strip()[:30]  # First 30 chars
                
                if line_num < len(pdf2_full_text):
                    # Get next line as context
                    next_line = pdf2_full_text[line_num] if line_num < len(pdf2_full_text) else ""
                    if next_line.strip():
                        context_after = next_line.strip()[:30]  # First 30 chars
                
                # Use a unique portion of the content
                if len(content) > 20:
                    search_text = content[:20]  # Use first 20 chars for uniqueness
                else:
                    search_text = content
                
                highlight_unique_text(search_text, [1, 1, 0], context_before, context_after)
        
        # Highlight changed lines (orange) - highlight only the changed words
        for diff in changed_lines:
            old_content = diff.get('old_content', '').strip()
            new_content = diff.get('new_content', '').strip()
            line_num = diff.get('doc2_line')
            
            if new_content and len(new_content) > 3:
                # Find specific words that changed
                changed_words = find_word_differences(old_content, new_content)
                
                if changed_words:
                    # Highlight each changed word group
                    for word_group in changed_words:
                        if len(word_group.strip()) > 2:  # Only meaningful words
                            # Get context from surrounding lines
                            context_before = ""
                            context_after = ""
                            
                            if line_num and line_num <= len(pdf2_full_text):
                                if line_num > 1:
                                    prev_line = pdf2_full_text[line_num - 2] if line_num > 1 else ""
                                    if prev_line.strip():
                                        context_before = prev_line.strip()[:30]
                                
                                if line_num < len(pdf2_full_text):
                                    next_line = pdf2_full_text[line_num] if line_num < len(pdf2_full_text) else ""
                                    if next_line.strip():
                                        context_after = next_line.strip()[:30]
                            
                            # Use the changed word group with context
                            highlight_unique_text(word_group, [1, 0.5, 0], context_before, context_after)
                else:
                    # Fallback: highlight the entire new content with context
                    if line_num and line_num <= len(pdf2_full_text):
                        context_before = ""
                        context_after = ""
                        if line_num > 1:
                            prev_line = pdf2_full_text[line_num - 2] if line_num > 1 else ""
                            if prev_line.strip():
                                context_before = prev_line.strip()[:30]
                        if line_num < len(pdf2_full_text):
                            next_line = pdf2_full_text[line_num] if line_num < len(pdf2_full_text) else ""
                            if next_line.strip():
                                context_after = next_line.strip()[:30]
                        
                        search_text = new_content[:30] if len(new_content) > 30 else new_content
                        highlight_unique_text(search_text, [1, 0.5, 0], context_before, context_after)
        
        # Save highlighted PDF
        doc.save(output_path)
        doc.close()
        
        if highlighted_count == 0:
            st.warning("⚠️ No text matches found for highlighting. The PDF text might be in images or have different formatting.")
        else:
            st.info(f"✅ Highlighted {highlighted_count} unique difference(s) in the PDF.")
        
        return True
        
    except Exception as e:
        st.error(f"Error creating highlighted PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
            st.info("ℹ️ **Note**: Full HTML and CSS support via headless browser. First-time use will install browser automatically (~100MB). Content between %%[ and ]%% is removed before conversion.")
            html_content = st.text_area(
                "HTML Content",
                key='html_content',
                height=300,
                help="Paste your HTML here. It will be converted to PDF for comparison. Any block that starts with %%[ and ends with ]%% is ignored. The entire content is rendered as a single page.",
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
                st.markdown("**HTML Preview:**")
                try:
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
                        
                        # Handle Document 2 - either PDF or HTML
                        if doc2_input_type == "Upload PDF":
                            pdf2_path = save_uploaded_file(pdf2_file, temp_dir)
                            pdf2_name = pdf2_file.name
                        else:
                            # Convert HTML to PDF
                            with st.spinner("Converting HTML to PDF..."):
                                pdf2_path = os.path.join(temp_dir, "html_converted.pdf")
                                if not html_to_pdf(html_content, pdf2_path):
                                    st.error("❌ Failed to convert HTML to PDF. Please check your HTML content.")
                                    st.stop()
                                pdf2_name = "HTML Document"
                                st.success("✅ HTML converted to PDF successfully!")
                        
                        # Initialize comparator (with caching)
                        if st.session_state.comparator is None:
                            with st.spinner("Loading Hugging Face model (first time only)..."):
                                st.session_state.comparator = PDFComparator()
                        
                        # Extract texts for comparison
                        text1 = st.session_state.comparator.extract_text_from_pdf(pdf1_path)
                        text2 = st.session_state.comparator.extract_text_from_pdf(pdf2_path)
                        
                        # Calculate similarity
                        semantic_sim_max, semantic_sim_avg = st.session_state.comparator.calculate_semantic_similarity(text1, text2)
                        
                        # Find differences
                        text_diff = st.session_state.comparator.find_text_differences(text1, text2)
                        
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
                    # Create highlighted version of PDF1 (shows removals in red)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        highlighted_pdf1_path = tmp_file.name
                        highlight_pdf_removals(
                            st.session_state.pdf1_path,
                            st.session_state.text_diff,
                            highlighted_pdf1_path
                        )
                        st.session_state.highlighted_pdf1_path = highlighted_pdf1_path
                    
                    # Create highlighted version of PDF2 (shows additions/changes)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        highlighted_pdf2_path = tmp_file.name
                        highlight_pdf_differences(
                            st.session_state.pdf2_path,
                            st.session_state.text_diff,
                            highlighted_pdf2_path
                        )
                        st.session_state.highlighted_pdf2_path = highlighted_pdf2_path
                    
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
            # Calculate semantic similarity
            try:
                text1 = st.session_state.comparator.extract_text_from_pdf(st.session_state.pdf1_path)
                text2 = st.session_state.comparator.extract_text_from_pdf(st.session_state.pdf2_path)
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
                        if st.session_state.get('highlighted_pdf2_path'):
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
                            st.image(page1_img, use_container_width=True, caption=f"Page {page_to_view} from Document 1 - Removed content highlighted in red")
                        else:
                            st.info("Page not available in Document 1")
                    
                    with col2:
                        st.markdown(f"### 📄 Document 2 - Page {page_to_view} (with highlights)")
                        if page2_img:
                            st.image(page2_img, use_container_width=True, caption=f"Page {page_to_view} from Document 2 - Differences highlighted")
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
