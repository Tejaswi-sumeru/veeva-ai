import re
import base64
import io
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup, NavigableString
from typing import List, Dict, Tuple, Optional
import difflib

def extract_html_text_map(html_content: str) -> List[Dict]:
    """
    Extracts visible text nodes from HTML and builds a map with metadata.
    Each entry contains: text, normalized text, and a reference for highlighting.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text_map = []
    def is_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        return True

    text_nodes = [node for node in soup.find_all(string=True) if is_visible(node)]
    
    for i, node in enumerate(text_nodes):
        original_text = str(node)
        if not original_text.strip():
            continue
            
        normalized = re.sub(r'\s+', ' ', original_text.strip().lower())
        text_map.append({
            'id': i,
            'node': node,
            'original': original_text,
            'normalized': normalized
        })
        
    return text_map, soup

def highlight_html_content(html_content: str, added_chunks: List[str]) -> str:
    """
    Advanced HTML highlighting:
    1. Linearize all visible text.
    2. Build a mapping between 'clean' text and raw NavigableStrings.
    3. Fuzzy search for semantic chunks in linearized text.
    4. Inject spans precisely, even across node boundaries.
    """
    from bs4 import BeautifulSoup, NavigableString
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. build raw offset map and full string
    raw_text = ""
    raw_map = [] # (node, offset_in_node) for each char in raw_text
    
    for node in soup.find_all(string=True):
        if node.parent.name in ['script', 'style', 'head', 'title', 'meta']:
            continue
        text = str(node)
        
        pos = 0
        while pos < len(text):
            match = re.search(r'%%.*?%%', text[pos:], flags=re.DOTALL)
            if match:
                start_before = pos
                end_before = pos + match.start()
                for i in range(start_before, end_before):
                    raw_map.append((node, i))
                    raw_text += text[i]
                pos += match.end()
            else:
                for i in range(pos, len(text)):
                    raw_map.append((node, i))
                    raw_text += text[i]
                break
            
    # 2. Build 'clean' string for searching
    clean_text = ""
    clean_to_raw = [] 
    
    for i, char in enumerate(raw_text):
        if char.isspace():
            if not clean_text or not clean_text[-1] == ' ':
                clean_text += ' '
                clean_to_raw.append(i)
        else:
            clean_text += char.lower()
            clean_to_raw.append(i)
            
    highlighted_mask = [False] * len(clean_text)
    sorted_chunks = sorted([c.strip() for c in added_chunks if len(c.strip()) > 5], key=len, reverse=True)
    
    for chunk in sorted_chunks:
        clean_chunk = re.sub(r'\s+', ' ', chunk.strip().lower())
        if not clean_chunk: continue
        
        start_pos = clean_text.find(clean_chunk)
        if start_pos == -1:
            matcher = difflib.SequenceMatcher(None, clean_chunk, clean_text)
            match = matcher.find_longest_match(0, len(clean_chunk), 0, len(clean_text))
            if match.size / len(clean_chunk) > 0.85:
                start_pos = match.b
                match_len = match.size
            else:
                u_chunk = "".join(c for c in clean_chunk if c.isalnum())
                u_text = ""
                u_map = []
                for idx, c in enumerate(clean_text):
                    if c.isalnum():
                        u_text += c
                        u_map.append(idx)
                
                u_start = u_text.find(u_chunk)
                if u_start != -1 and len(u_chunk) > 10:
                    start_pos = u_map[u_start]
                    match_len = u_map[u_start + len(u_chunk) - 1] - start_pos + 1
                else:
                    continue
        else:
            match_len = len(clean_chunk)
            
        for i in range(start_pos, start_pos + match_len):
            if i < len(highlighted_mask):
                highlighted_mask[i] = True

    # 3. GAP FILLING (merge adjacent highlights): Fill gaps between highlighted segments so
    # "Word 1, Word 2" becomes one green block. Fill gap (set to True) if:
    #   - gap is shorter than 12 characters, OR
    #   - gap consists only of whitespace and punctuation.
    new_mask = list(highlighted_mask)
    i = 0
    while i < len(highlighted_mask):
        if highlighted_mask[i]:
            next_h = -1
            for j in range(i + 1, len(highlighted_mask)):
                if highlighted_mask[j]:
                    next_h = j
                    break
            if next_h != -1:
                gap_len = next_h - i - 1
                if gap_len > 0:
                    gap_text = clean_text[i + 1 : next_h]
                    is_whitespace_and_punctuation = not any(c.isalnum() for c in gap_text)
                    if gap_len < 12 or is_whitespace_and_punctuation:
                        for k in range(i + 1, next_h):
                            new_mask[k] = True
            i = max(i + 1, next_h if next_h != -1 else len(highlighted_mask))
        else:
            i += 1
    highlighted_mask = new_mask

    # 4. Contiguous raw ranges
    raw_highlights = [False] * len(raw_text)
    for i, val in enumerate(highlighted_mask):
        if val:
            raw_idx = clean_to_raw[i]
            raw_highlights[raw_idx] = True

    ranges = []
    start = -1
    for i in range(len(raw_highlights)):
        if raw_highlights[i] and start == -1:
            start = i
        elif not raw_highlights[i] and start != -1:
            ranges.append((start, i))
            start = -1
    if start != -1:
        ranges.append((start, len(raw_highlights)))
        
    from collections import defaultdict
    node_to_ranges = defaultdict(list)
    
    for r_start, r_end in ranges:
        for i in range(r_start, r_end):
            node, node_off = raw_map[i]
            if not node_to_ranges[node] or node_to_ranges[node][-1][1] < node_off:
                node_to_ranges[node].append([node_off, node_off + 1])
            else:
                node_to_ranges[node][-1][1] = node_off + 1

    for node, fragments in node_to_ranges.items():
        if not node.parent: continue
        text = str(node)
        current_node = node
        for start, end in sorted(fragments, key=lambda x: x[0], reverse=True):
            before_text = text[:start]
            highlight_text = text[start:end]
            after_text = text[end:]
            span = soup.new_tag("span", **{
                "class": "diff-added",
                "style": "background-color: #d4edda; border-bottom: 2px solid #28a745; border-radius: 2px; -webkit-print-color-adjust: exact;"
            })
            span.string = highlight_text
            if after_text:
                current_node.insert_after(NavigableString(after_text))
            current_node.insert_after(span)
            if before_text:
                new_before = NavigableString(before_text)
                current_node.replace_with(new_before)
                current_node = new_before
                text = before_text 
            else:
                current_node.extract()
                break
                
    return str(soup)

def normalize_text_semantic(text: str) -> str:
    """Normalize text for semantic comparison."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def check_litmus_tracking(html_content: str) -> bool:
    """
    Checks if Litmus tracking code is present in the HTML.
    Looks for the 'emltrk.com' domain or standard Litmus patterns.
    """
    if not html_content:
        return False
    return 'emltrk.com' in html_content

def check_image_alt_matches_link_alias(html_content: str) -> List[str]:
    """
    Checks if images inside links have an alt text that matches the link's alias.
    According to the rule: "Alt text and Link alias name should match"
    
    Returns a list of error messages for non-compliant tags.
    """
    if not html_content:
        return []
        
    soup = BeautifulSoup(html_content, 'html.parser')
    errors = []
    def clean_str(s):
        s = re.sub(r'%%=v\(@[^)]+\)=%%', '', s)
        s = re.sub(r'[_\-]+', ' ', s)
        return s.strip().lower()

    for a_tag in soup.find_all('a'):
        alias = a_tag.get('alias', '')
        if not alias:
            continue
        for img in a_tag.find_all('img'):
            alt = img.get('alt', '')
            if not alt:
                continue
            c_alias = clean_str(alias)
            c_alt = clean_str(alt)
            if c_alt not in c_alias:
                if alt.lower() not in alias.lower():
                    errors.append(f"⚠️ Mismatch: Link alias '{alias}' does not contain Image alt '{alt}'")
    return errors


def check_alias_links_img_has_alt(html_content: str) -> List[str]:
    """For every <a> with alias that contains an <img>, check that the image has an alt attribute (presence only)."""
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    errors = []
    for a_tag in soup.find_all('a'):
        alias = a_tag.get('alias', '').strip()
        if not alias:
            continue
        for img in a_tag.find_all('img'):
            if not img.get('alt', '').strip():
                errors.append(f"❌ Image missing alt (Link alias: {alias})")
    return errors

def check_missing_title_attributes(html_content: str) -> List[str]:
    """Checks that every <a> with an alias has a title attribute (link only; images are not checked)."""
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    errors = []
    for a in soup.find_all('a'):
        alias = a.get('alias', '').strip()
        if not alias:
            continue
        if not a.get('title', '').strip():
            errors.append(f"❌ Link missing title (Alias: {alias})")
    return errors


def _normalize_url_for_link_check(uri: str) -> str:
    """Must match compare_pdfs._normalize_url_for_link_check for PDF/HTML comparison."""
    if not uri or not isinstance(uri, str):
        return ""
    u = uri.strip().lower()
    if "#" in u:
        u = u.split("#")[0]
    u = u.rstrip("/") or "/"
    return u


def _get_gemini_api_key_from_streamlit_secrets() -> Optional[str]:
    """Read Gemini/Google API key from Streamlit secrets if running in Streamlit. Otherwise return None."""
    try:
        import streamlit as st
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        return secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _llm_link_metadata_consistent(url: str, metadata_str: str) -> Optional[bool]:
    """
    Use Gemini 2.5 Flash to decide if link metadata (title, alt, text) is consistent with the URL.
    Returns True if consistent, False if not, None if LLM unavailable (no API key or import error).
    """
    try:
        import os
        from google import genai
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or _get_gemini_api_key_from_streamlit_secrets()
        )
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        prompt = (
            "Given this link URL and the metadata (title, alt text, link text) shown for it, "
            "is the metadata consistent with the URL? "
            "E.g. App Store URL should have App Store/Apple-like metadata; Google Play URL should have Google Play-like metadata. "
            "Answer with exactly one word: YES or NO.\n\n"
            f"URL: {url}\nMetadata: {metadata_str or '(none)'}"
        )
        print("[DEBUG] Making LLM call (link metadata consistency)...", flush=True)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = (response.text or "").strip().upper()
        if "YES" in answer:
            return True
        if "NO" in answer:
            return False
        return None
    except Exception as e:
        print(f"[DEBUG] LLM call error (link metadata check): {e!r}", flush=True)
        return None


def _extract_html_links_with_metadata(html_content: str) -> List[Dict]:
    """Extract all <a href="..."> with href, title, alt (from child img), link text, alias."""
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        title = (a.get("title") or "").strip()
        alias = (a.get("alias") or "").strip()
        alts = []
        for img in a.find_all("img"):
            alt = (img.get("alt") or "").strip()
            if alt:
                alts.append(alt)
        text = (a.get_text(separator=" ", strip=True) or "").strip()
        metadata_parts = [title, alias, text] + alts
        metadata_str = " ".join(p for p in metadata_parts if p).lower()
        links.append({
            "href": href,
            "href_normalized": _normalize_url_for_link_check(href),
            "title": title,
            "alt": " ".join(alts),
            "text": text,
            "metadata": metadata_str,
        })
    return links


def check_links_against_pdf(
    pdf_urls: List[str],
    html_content: str,
) -> Dict[str, List[str]]:
    """
    Verify HTML links against PDF: (1) each link URL must be in PDF; (2) link metadata should match URL.
    pdf_urls: list of normalized URLs from the PDF (from compare_pdfs.extract_pdf_link_urls).
    Returns dict with keys: "not_in_pdf", "metadata_mismatch", "valid_links".
    - not_in_pdf: messages "The approved PDF does not contain this link." with URL
    - metadata_mismatch: messages "The link does not seem correct." with URL/details
    - valid_links: list of display strings for links that passed both checks
    """
    result = {"not_in_pdf": [], "metadata_mismatch": [], "valid_links": []}
    pdf_set = set(pdf_urls or [])
    html_links = _extract_html_links_with_metadata(html_content or "")
    for link in html_links:
        href_norm = link["href_normalized"]
        if not href_norm:
            continue
        if href_norm not in pdf_set:
            result["not_in_pdf"].append(
                f"The approved PDF does not contain this link.\n{link['href']}"
            )
            continue
        consistent = _llm_link_metadata_consistent(link["href"], link["metadata"])
        if consistent is False:
            result["metadata_mismatch"].append(
                f"The link does not seem correct.\n{link['href']}"
                + (f" (title/alt/text: {link['title'] or link['alt'] or link['text'] or '(none)'})" if (link.get("title") or link.get("alt") or link.get("text")) else "")
            )
            continue
        # In PDF and metadata consistent (or LLM unavailable)
        label = link.get("title") or link.get("alias") or link.get("text") or link["href"]
        result["valid_links"].append(f"{label}\n{link['href']}")
    return result


def check_sumeru_links(html_content: str) -> List[str]:
    """
    Check that no image or link URL in HTML contains 'Sumeru' (case-insensitive).
    Flags any Sumeru link: in buttons, embedded, from CDN, or any other source.
    Returns list of offending URLs; empty if none.
    """
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, "html.parser")
    found = []
    for tag, attr in [("img", "src"), ("a", "href")]:
        for el in soup.find_all(tag, **{attr: True}):
            url = (el.get(attr) or "").strip()
            if url and "sumeru" in url.lower():
                found.append(url)
    return found


# Tokens that suggest an image is a header/logo (alt, class, id)
_HEADER_LOGO_TOKENS = ("logo", "header", "brand")


def _is_header_logo_image(img) -> bool:
    """True if img looks like a header/logo (alt, class, or id contains logo/header/brand)."""
    alt = (img.get("alt") or "").strip().lower()
    cls = (img.get("class") or [])
    if isinstance(cls, str):
        cls = [cls]
    class_str = " ".join(cls).lower()
    id_attr = (img.get("id") or "").strip().lower()
    combined = f"{alt} {class_str} {id_attr}"
    return any(t in combined for t in _HEADER_LOGO_TOKENS)


def _href_is_valid(href: str) -> bool:
    """True if href looks like a real link (not # or javascript:void(0))."""
    if not href or not href.strip():
        return False
    h = href.strip().lower()
    if h in ("#", ""):
        return False
    if h.startswith("javascript:"):
        return False
    return True


def check_header_logo_clickable(html_content: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Check that every header/logo image (alt, class, or id contains logo/header/brand)
    is wrapped in an <a href="..."> with a valid destination.
    Returns: {"not_clickable": [{"alt": str, "src": str, "reason": str}], "all_clickable": bool}.
    """
    result = {"not_clickable": [], "all_clickable": True}
    if not html_content:
        return result
    soup = BeautifulSoup(html_content, "html.parser")
    for img in soup.find_all("img"):
        if not _is_header_logo_image(img):
            continue
        alt = (img.get("alt") or "").strip() or "(no alt)"
        src = (img.get("src") or "").strip()[:80]
        parent = img.parent
        link = None
        while parent and parent.name != "body":
            if parent.name == "a":
                link = parent
                break
            parent = getattr(parent, "parent", None)
        if link is None:
            result["all_clickable"] = False
            result["not_clickable"].append({"alt": alt, "src": src, "reason": "Not wrapped in a link"})
            continue
        href = (link.get("href") or "").strip()
        if not _href_is_valid(href):
            result["all_clickable"] = False
            result["not_clickable"].append({"alt": alt, "src": src, "reason": "Link has no valid href"})
    return result


def _parse_utm_params(utm_string_or_url: str) -> Dict[str, str]:
    """
    Parse a reference UTM string (e.g. ?utm_source=MFS&utm_medium=email&...)
    or a full URL into a flat dict of UTM param name -> value.
    """
    if not (utm_string_or_url or "").strip():
        return {}
    s = utm_string_or_url.strip()
    if "?" in s and ("http://" in s or "https://" in s):
        s = urlparse(s).query
    if s.startswith("?"):
        s = s[1:]
    parsed = parse_qs(s)
    return {k: (v[0] if v else "") for k, v in parsed.items() if k.startswith("utm_")}


def verify_utm_in_internal_links(
    html_content: str,
    reference_utm_string: str,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Check that every internal link (http/https) in HTML has UTM params that match
    the reference. Reference is parsed from reference_utm_string (query string or full URL).
    Returns dict: "mismatch" = list of {"url": str, "reason": str}; "all_match" = bool.
    """
    expected = _parse_utm_params(reference_utm_string)
    if not expected:
        return {"mismatch": [], "all_match": True, "message": "No UTM params in reference."}
    result = {"mismatch": [], "all_match": True}
    if not html_content:
        return result
    soup = BeautifulSoup(html_content, "html.parser")
    seen_urls = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or not (href.startswith("http://") or href.startswith("https://")):
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        parsed = urlparse(href)
        link_params = _parse_utm_params(parsed.query or "")
        reason_parts = []
        for key, expected_val in expected.items():
            if key not in link_params:
                reason_parts.append(f"missing {key}")
            elif link_params[key] != expected_val:
                reason_parts.append(f"{key}=... (expected {expected_val!r}, got {link_params[key]!r})")
        if reason_parts:
            result["all_match"] = False
            result["mismatch"].append({"url": href, "reason": "; ".join(reason_parts)})
    return result


def _normalize_phone(s: str) -> str:
    """Normalize phone to digits only (with optional leading +)."""
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    return digits


def extract_phone_numbers_from_html(html_content: str) -> List[str]:
    """
    Extract phone numbers from HTML: AMPscript RedirectTo('tel:...') and href="tel:...".
    Returns list of normalized phone strings (digits only) for comparison.
    """
    if not html_content:
        return []
    numbers = []
    # %%=RedirectTo('tel:+18443305535')=%%
    for m in re.finditer(r"RedirectTo\s*\(\s*['\"]tel:([^'\"]+)['\"]", html_content, re.I):
        numbers.append(_normalize_phone(m.group(1)))
    # href="tel:+1..."
    for m in re.finditer(r"href\s*=\s*['\"]tel:([^'\"]+)['\"]", html_content, re.I):
        numbers.append(_normalize_phone(m.group(1)))
    return [n for n in numbers if len(n) >= 10]


def check_phone_numbers_against_pdf(html_content: str, pdf_text: str) -> Dict[str, List[str]]:
    """
    Check that every phone number found in HTML appears in the PDF text.
    pdf_text: raw text extracted from the approved PDF.
    Returns dict: "missing_in_pdf" = list of phone numbers (or messages) not found in PDF; "all_found" = bool.
    """
    html_phones = list(dict.fromkeys(extract_phone_numbers_from_html(html_content)))  # dedupe, preserve order
    if not html_phones:
        return {"missing_in_pdf": [], "all_found": True}
    pdf_norm = _normalize_phone(pdf_text)
    missing = []
    for p in html_phones:
        if p not in pdf_norm:
            missing.append(p)
    return {"missing_in_pdf": missing, "all_found": len(missing) == 0}


def check_email_image_quality(html_content: str) -> List[str]:
    """Classify each image by display size (icon/content/hero) and run type-specific quality checks. Returns list of error strings."""
    details = check_email_image_quality_with_details(html_content)
    return [d["message"] for d in details]


def check_email_image_quality_with_details(html_content: str) -> List[dict]:
    """Same as check_email_image_quality but returns list of dicts with message, alt, image_bytes (for optional thumbnail display)."""
    try:
        from email_image_quality import validate_html_images_with_details
        return validate_html_images_with_details(html_content, skip_src_pattern="emltrk.com")
    except Exception as e:
        return [{"message": f"⚠️ Image quality check failed: {e}", "alt": "", "image_bytes": None}]


