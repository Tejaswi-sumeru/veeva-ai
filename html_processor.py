import re
import base64
import io
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup, NavigableString
from typing import List, Dict, Tuple, Optional, Any
import difflib

# Footer = "Important Safety Information" and everything below it. Used to exclude from PDF comparison and for footer-vs-standard check.
FOOTER_ANCHOR_TEXT = "Important Safety Information"
FOOTER_START_MARKER_TAG = "<b>Important Safety Information</b>"
FOOTER_START_MARKER_COMMENT = "<!-- Section footer -->"


def _find_footer_start_index(html_content: str) -> int:
    """
    Return the index in html_content where the footer starts (exact area: Important Safety Information and below).
    Snaps to the start of the tag containing that text so we don't cut mid-element.
    Returns -1 if not found.
    """
    # Prefer exact tag match (standard HTML)
    idx = html_content.find(FOOTER_START_MARKER_TAG)
    if idx != -1:
        return idx
    # Snap to "Important Safety Information" and below: find text then start of containing tag
    idx = html_content.find(FOOTER_ANCHOR_TEXT)
    if idx != -1:
        tag_start = html_content.rfind("<", 0, idx)
        return tag_start if tag_start != -1 else idx
    # Fallback: comment that often precedes this section
    idx = html_content.find(FOOTER_START_MARKER_COMMENT)
    return idx


def get_html_without_footer(html_content: str) -> str:
    """
    Return HTML up to (but not including) the footer section.
    Footer is defined as "Important Safety Information" and everything below it.
    """
    if not html_content:
        return html_content
    start = _find_footer_start_index(html_content)
    if start != -1:
        return html_content[:start].rstrip()
    return html_content


def extract_footer_from_html(html_content: str) -> str:
    """Return the footer portion: from 'Important Safety Information' (and below) to end."""
    if not html_content:
        return ""
    start = _find_footer_start_index(html_content)
    if start != -1:
        return html_content[start:].strip()
    return ""


def _normalize_footer_text(html_fragment: str) -> str:
    """Normalize footer HTML to comparable text (strip tags, collapse whitespace)."""
    if not html_fragment:
        return ""
    soup = BeautifulSoup(html_fragment, "html.parser")
    text = (soup.get_text() or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _footer_diff_highlight_html(std_text: str, user_text: str) -> Tuple[str, str]:
    """
    Return (standard_html, user_html) with differing regions wrapped in spans for display.
    Uses word-level diff so the actual changed words are highlighted.
    """
    import html as html_module
    words_std = std_text.split()
    words_user = user_text.split()
    matcher = difflib.SequenceMatcher(None, words_std, words_user)
    std_parts = []
    user_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            std_parts.append(html_module.escape(" ".join(words_std[i1:i2])))
            user_parts.append(html_module.escape(" ".join(words_user[j1:j2])))
        elif tag == "replace":
            std_parts.append('<span style="background:#ffcccc;padding:0 2px;">' + html_module.escape(" ".join(words_std[i1:i2])) + "</span>")
            user_parts.append('<span style="background:#ccffcc;padding:0 2px;">' + html_module.escape(" ".join(words_user[j1:j2])) + "</span>")
        elif tag == "delete":
            std_parts.append('<span style="background:#ffcccc;padding:0 2px;">' + html_module.escape(" ".join(words_std[i1:i2])) + "</span>")
        elif tag == "insert":
            user_parts.append('<span style="background:#ccffcc;padding:0 2px;">' + html_module.escape(" ".join(words_user[j1:j2])) + "</span>")
    return (" ".join(std_parts), " ".join(user_parts))


def compare_footer_to_standard(html_content: str, standard_footer_html: str) -> Dict[str, Any]:
    """
    Compare the footer in html_content to the standard footer.
    Returns: {"match": bool, "differences": list of str, "user_footer_text": str, "standard_footer_text": str,
              "user_footer_highlighted_html": str, "standard_footer_highlighted_html": str}.
    """
    user_footer = extract_footer_from_html(html_content or "")
    user_text = _normalize_footer_text(user_footer)
    standard_text = _normalize_footer_text(standard_footer_html or "")
    if not user_footer and not standard_footer_html:
        return {"match": True, "differences": [], "user_footer_text": "", "standard_footer_text": "",
                "user_footer_highlighted_html": "", "standard_footer_highlighted_html": ""}
    if not user_footer:
        return {"match": False, "differences": ["No footer found in pasted HTML."], "user_footer_text": "", "standard_footer_text": standard_text,
                "user_footer_highlighted_html": "", "standard_footer_highlighted_html": ""}
    if not standard_footer_html:
        return {"match": True, "differences": [], "user_footer_text": user_text, "standard_footer_text": "",
                "user_footer_highlighted_html": "", "standard_footer_highlighted_html": ""}
    # Word-level comparison so we see actual changes (e.g. v1.0 vs v2.0), not truncated long lines
    words_std = standard_text.split()
    words_user = user_text.split()
    matcher = difflib.SequenceMatcher(None, words_std, words_user)
    differences = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            expected = " ".join(words_std[i1:i2])
            got = " ".join(words_user[j1:j2])
            differences.append(f"Standard (reference) has: \"{expected}\" → Current has: \"{got}\"")
        elif tag == "delete":
            differences.append(f"Missing in current HTML: \"{' '.join(words_std[i1:i2])}\"")
        elif tag == "insert":
            differences.append(f"Extra in current HTML: \"{' '.join(words_user[j1:j2])}\"")
    if not differences and user_text.strip() != standard_text.strip():
        differences.append("Footer text differs from standard (structure or wording).")
    std_highlighted, user_highlighted = _footer_diff_highlight_html(standard_text, user_text)
    return {
        "match": len(differences) == 0,
        "differences": differences[:50],
        "user_footer_text": user_text[:2000],
        "standard_footer_text": standard_text[:2000],
        "user_footer_highlighted_html": user_highlighted,
        "standard_footer_highlighted_html": std_highlighted,
    }


# Standard (reference) footer HTML for comparison. Update this to match approved footer.
STANDARD_FOOTER_HTML = """
<tr>
      <td style="border-top:solid 2px #D9D9D6;">
      </td></tr><tr>
      <td height="20" style="height:20px;">
      </td></tr><tr>
      <td align="center" style="font-size:10px;font-weight:normal;line-height:14px;text-align:left;color:#383d48; font-family:Arial, Roboto, 'sans-serif',  Helvetica;">
        <b>Important Safety Information</b></td></tr><tr>
      <td height="4" style="height:4px;">
      </td></tr><tr>
      <td align="center" style="font-size:10px;font-weight:normal;line-height:14px;text-align:left;color:#383d48; font-family:Arial, Roboto, 'sans-serif',  Helvetica;">
        Product for prescription only, for Important Safety Information please visit <a alias="FSL3Plus_footer-isi-v2_121115" conversion="false" data-linkto="http://" href="https://www.sumeru.us/" style="color:#031c8d;text-decoration:underline;" title="FreeStyleLibre.us">FreeStyleLibre.us</a></td></tr><tr>
      <td height="4" style="height:4px;">
      </td></tr><tr>
      <td align="center" style="font-size:10px;font-weight:normal;line-height:14px;text-align:left;color:#383d48; font-family:Arial, Roboto, 'sans-serif',  Helvetica;">
        The sensor housing, FreeStyle, Libre, and related brand marks are marks of Abbott. Other trademarks are the property of their respective owners.</td></tr><tr>
      <td height="6" style="height:6px;">
      </td></tr><tr>
      <td align="center" style="font-size:10px;font-weight:normal;line-height:14px;text-align:left;color:#383d48; font-family:Arial, Roboto, 'sans-serif',  Helvetica;">
        ADC-88084 v3.0</td></tr><tr>
      <td align="center" height="20">
      </td></tr><tr>
      <td style="border-top:solid 2px #D9D9D6;">
      </td></tr></table></td></tr></table><table cellpadding="0" cellspacing="0" width="100%" role="presentation" style="min-width: 100%; " class="stylingblock-content-wrapper"><tr><td class="stylingblock-content-wrapper camarker-inner"><!-- Section footer --><table align="center" bgcolor="#ffffff" border="0" cellpadding="0" cellspacing="0" class="w100" role="presentation" style="width:100% background-color:#ffffff; margin:0 auto; color:#000000;" width="100%">
    <tr>
      <td align="center" height="20" style="height:20px;">
        &nbsp;</td></tr><tr>
      <td align="center" class="mobtxt" style="text-align:center; padding:0; color: #6D7D8B; font-family:arial,helvetica,sans-serif; font-size:16px; line-height:normal; font-weight:bold; color:#383d48">
        Follow us for more healthy inspiration!</td></tr><tr>
      <td align="center" height="26" style="height:26px;">
        &nbsp;</td></tr><tr>
      <td align="center" style="padding:0px 50px 0px 50px;">
        <table align="center" border="0" cellpadding="0" cellspacing="0" class="widthIcons" style="width:140px;" width="140">
            <tr>
              <td class="w10" style="width:3px;" width="3">
              </td><td align="left">
                <div>
                  <a alias="GDI_Support_FSL3Plus_facebook_121115" conversion="false" data-linkto="https://" href="https://www.facebook.com/FreeStyleDiabetes/" target="_blank" title="Facebook"><img alt="Facebook" data-assetid="881557" height="38" src="http://image.freestyle.abbott.us/lib/fe3011717d64047f701076/m/1/955d48b3-cb80-47f9-9d29-655ae78b8e75.png" style="display: block; width: 38px; max-width: 38px; padding: 0px; height: 38px; text-align: center;" width="38"></a></div></td><td class="w10" style="width:3px;" width="3">
              </td><td align="left">
                <div>
                  <a alias="GDI_Support_FSL3Plus_instagram_121115" conversion="false" data-linkto="https://" href="https://www.instagram.com/freestylediabetes/" target="_blank" title="Instagram"><img alt="Instagram" data-assetid="881558" height="38" src="http://image.freestyle.abbott.us/lib/fe3011717d64047f701076/m/1/5d8ed3f8-712e-43f1-bbf1-1ac9d7b48ebf.png" style="display: block; width: 38px; max-width: 38px; padding: 0px; text-align: center; height: 38px;" width="38"></a></div></td><td class="w10" style="width:3px;" width="3">
              </td><td align="left">
                <div>
                  <a alias="GDI_Support_FSL3Plus_youtube_121115" conversion="false" data-linkto="https://" href="https://www.youtube.com/freestyleus" target="_blank" title="Youtube"><img alt="Youtube" data-assetid="881559" height="38" src="http://image.freestyle.abbott.us/lib/fe3011717d64047f701076/m/1/685372cb-9423-41e4-8cdc-774186e5c651.png" style="display: block; width: 38px; max-width: 38px; padding: 0px; text-align: center; height: 38px;" width="38"></a></div></td></tr></table></td></tr><tr>
      <td align="center" height="26" style="height:26px;">
        &nbsp;</td></tr><tr>
      <td align="center">
        <table align="center" bgcolor="#ffffff" border="0" cellpadding="0" cellspacing="0" class="w100" role="presentation" style="width:40% background-color:#ffffff; margin:0 auto; color:#000000;" width="40%">
            <tr>
              <td style="border-top:solid 1px #d8d8d6;">
                &nbsp;</td></tr></table></td></tr><tr>
      <td align="center" style="">
        <a alias="GDI_Support_FSL3Plus_footer-abbott-logo_121115" conversion="false" data-linkto="http://" href="http://www.abbott.com/corpnewsroom.html?utm_source=MFS&utm_medium=email&utm_campaign=GDI&utm_content=121115" title="Abbott"><img alt="Abbott Logo" data-assetid="844357" height="69" src="https://image.freestyle.abbott.us/lib/fe2f11717d64047d701277/m/1/abbott_logo.png" style="height: 69px; width: 134px; padding: 0px; text-align: center;" width="134"></a></td></tr><tr>
      <td align="center" height="20" style="height:20px;">
        &nbsp;</td></tr><tr>
      <td align="center" class="mobtxt" style="text-align:center; padding:0px 50px 0 50px; color: #383d48; font-family: 'Roboto', sans-serif; font-size:12px; line-height:24px; font-weight:400; ">
        <table align="center" bgcolor="#ffffff" border="0" cellpadding="0" cellspacing="0" class="w100" role="presentation" style="width:70% background-color:#ffffff; margin:0 auto; color:#000000;" width="70%">
            <tr>
              <th class="centered logo" style="text-align:right; color: #383d48; font-family: 'Roboto', sans-serif; font-size:12px; line-height:normal; font-weight:400; margin:0; padding:0;">
                Abbott Laboratories,</th><th class="centered logo" style="text-align:center; color: #383d48; font-family: 'Roboto', sans-serif; font-size:12px; line-height:normal; font-weight:400; margin:0; padding:0;">
                <a style="text-decoration:none; color:#383d48;"> 100 Abbott Park Road,</a></th><th class="centered logo" style="text-align:left; color: #383d48; font-family: 'Roboto', sans-serif; font-size:12px; line-height:normal; font-weight:400; margin:0; padding:0;">
                <a style="text-decoration:none; color:#383d48;"> Abbott Park, IL 60064 </a></th></tr></table></td></tr><tr>
      <td class="mobtxt" style="text-align:center; padding:0px 50px 0 50px; color: #383D48; font-family: 'Roboto', sans-serif; font-size:12px; line-height:24px; font-weight:400; ">
        &copy; 2026 Abbott. All rights reserved. ADC-88080 v1.0</td></tr><tr>
      <td align="center" height="20" style="height:20px;">
        &nbsp;</td></tr><tr>
      <td align="center" class="mobtxt" style="text-align:center; padding:0px 50px 0 50px; color: #383d48; font-family:arial,helvetica,sans-serif; font-size:12px; line-height:24px; font-weight:400; ">
        <a alias="GDI_Support_FSL3Plus_unsubscribe_121115" conversion="false" data-linkto="other" href="https://mcp0vmry3pl80dzlgp450h5p5dx0.pub.sfmc-content.com/rlfffjufqnp?qs=eyJkZWtJZCI6ImEzOWMwMTlkLTE2ZjQtNGVjMi1hNzdiLTY0NjM3YjRmMDNiOCIsImRla1ZlcnNpb24iOjEsIml2IjoiZGFJdzhXZHlOb0FZZzhiaXpSUlVlZz09IiwiY2lwaGVyVGV4dCI6IjV5TE1oc0ZCQlE3UERWN0NvZGdTTGllVlhOWWlYK2VMZ3phaThQSE1pT1c1VWJsOHVteFpQeElDa2FHZmplbDFFS09ueFRNNWFjVk9iVEFWc0NIMzRUcHU2VWgrYWY3dENHeDVDM1NqT2I3cGJqaEs1U2lhV1pBRzNKQ2JkYUl3OFdkeU5vQVlnOGJpelJSVWVnPT0iLCJhdXRoVGFnIjoib3ptKzZXNDRTdVVvbWxtUUJ0eVFtdz09In0%3D" style="color:#383d48;text-decoration:underline;" title="Unsubscribe">Unsubscribe</a></td></tr><tr>
      <td align="center" height="26" style="height:26px; border-bottom:solid 12px #031c8d;">
        &nbsp;</td></tr></table></td></tr></table>
"""

# Expected social media URLs in the footer (from standard footer). Used to verify current HTML footer links.
STANDARD_FOOTER_SOCIAL_URLS = {
    "Facebook": "https://www.facebook.com/FreeStyleDiabetes/",
    "Instagram": "https://www.instagram.com/freestylediabetes/",
    "Youtube": "https://www.youtube.com/freestyleus",
}


def _normalize_url_for_compare(url: str) -> str:
    """Normalize URL for comparison (lower, strip trailing slash)."""
    if not url:
        return ""
    u = url.strip().lower()
    return u.rstrip("/") if u != "/" else u


def check_footer_social_links(html_content: str) -> Dict[str, Any]:
    """
    Verify that social media icon links in the footer match the standard footer.
    Only checks links in the footer section (Important Safety Information and below).
    Returns: {"all_match": bool, "missing": [platform], "mismatch": [{"platform": str, "expected": str, "got": str}]}.
    """
    result = {"all_match": True, "missing": [], "mismatch": []}
    footer_html = extract_footer_from_html(html_content or "")
    if not footer_html:
        return result
    soup = BeautifulSoup(footer_html, "html.parser")
    # Find links that wrap an img with alt/title Facebook, Instagram, or Youtube
    platform_keys = {"facebook": "Facebook", "instagram": "Instagram", "youtube": "Youtube"}
    found = {}  # platform -> href
    for a in soup.find_all("a", href=True):
        for img in a.find_all("img"):
            alt = (img.get("alt") or "").strip().lower()
            title = (img.get("title") or "").strip().lower()
            for key, label in platform_keys.items():
                if key in alt or key in title:
                    found[label] = (a.get("href") or "").strip()
                    break
    expected_norm = {p: _normalize_url_for_compare(u) for p, u in STANDARD_FOOTER_SOCIAL_URLS.items()}
    for platform, expected_url in STANDARD_FOOTER_SOCIAL_URLS.items():
        got = found.get(platform)
        if not got:
            result["all_match"] = False
            result["missing"].append(platform)
            continue
        if _normalize_url_for_compare(got) != expected_norm[platform]:
            result["all_match"] = False
            result["mismatch"].append({
                "platform": platform,
                "expected": expected_url,
                "got": got,
            })
    return result

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


async def _llm_link_metadata_consistent_async(url: str, metadata_str: str) -> Optional[bool]:
    """
    Async: use Gemini 2.5 Flash (aio) to decide if link metadata is consistent with the URL.
    Returns True if consistent, False if not, None if LLM unavailable.
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
        response = await client.aio.models.generate_content(
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


async def _check_links_against_pdf_async(
    pdf_urls: List[str],
    html_content: str,
) -> Dict[str, List[str]]:
    """Async: verify HTML links against PDF; run all Gemini (aio) calls concurrently."""
    import asyncio
    result = {"not_in_pdf": [], "metadata_mismatch": [], "valid_links": []}
    pdf_set = set(pdf_urls or [])
    html_links = _extract_html_links_with_metadata(html_content or "")
    links_in_pdf = []
    for link in html_links:
        href_norm = link["href_normalized"]
        if not href_norm:
            continue
        if href_norm not in pdf_set:
            result["not_in_pdf"].append(
                f"The approved PDF does not contain this link.\n{link['href']}"
            )
            continue
        links_in_pdf.append(link)
    if not links_in_pdf:
        return result
    coros = [_llm_link_metadata_consistent_async(link["href"], link["metadata"]) for link in links_in_pdf]
    consistencies = await asyncio.gather(*coros, return_exceptions=True)
    for link, consistent in zip(links_in_pdf, consistencies):
        if isinstance(consistent, Exception):
            continue
        if consistent is False:
            result["metadata_mismatch"].append(
                f"The link does not seem correct.\n{link['href']}"
                + (f" (title/alt/text: {link['title'] or link['alt'] or link['text'] or '(none)'})" if (link.get("title") or link.get("alt") or link.get("text")) else "")
            )
            continue
        label = link.get("title") or link.get("alias") or link.get("text") or link["href"]
        result["valid_links"].append(f"{label}\n{link['href']}")
    return result


def check_links_against_pdf(
    pdf_urls: List[str],
    html_content: str,
) -> Dict[str, List[str]]:
    """
    Verify HTML links against PDF: (1) each link URL must be in PDF; (2) link metadata should match URL.
    Uses async Gemini (aio) and runs all LLM calls concurrently via asyncio.
    Returns dict with keys: "not_in_pdf", "metadata_mismatch", "valid_links".
    """
    import asyncio
    return asyncio.run(_check_links_against_pdf_async(pdf_urls, html_content))


def check_sumeru_links(html_content: str) -> Dict[str, Any]:
    """
    Check that no image or link URL contains 'Sumeru', and that 'Sumeru' does not
    appear as plain text (non-clickable) in the content.
    Returns dict: "urls" = list of offending URLs; "plain_text_found" = True if Sumeru appears as plain text.
    """
    result = {"urls": [], "plain_text_found": False}
    if not html_content:
        return result
    soup = BeautifulSoup(html_content, "html.parser")
    for tag, attr in [("img", "src"), ("a", "href")]:
        for el in soup.find_all(tag, **{attr: True}):
            url = (el.get(attr) or "").strip()
            if url and "sumeru" in url.lower():
                result["urls"].append(url)
    # Strip script/style so we only check visible content
    for tag in soup(["script", "style"]):
        tag.decompose()
    visible_text = (soup.get_text() or "").lower()
    if "sumeru" in visible_text:
        result["plain_text_found"] = True
    return result


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


def _phone_found_in_text(phone_digits: str, pdf_digits: str) -> bool:
    """True if phone appears in PDF text. Handles +1: 11-digit (1XXXXXXXXXX) matches PDF with 10-digit (XXXXXXXXXX).
    Input is normalized to digits only so '+1' or spaces don't affect length (e.g. '+18443305535' -> 11 digits)."""
    p = _normalize_phone(phone_digits)
    if not p or len(p) < 10:
        return False
    if p in pdf_digits:
        return True
    if len(p) == 11 and p.startswith("1"):
        if p[1:] in pdf_digits:
            return True
    return False

def check_phone_numbers_against_pdf(html_content: str, pdf_text: str) -> Dict[str, List[str]]:
    """
    Check that every phone number found in HTML appears in the PDF text.
    Handles country code: HTML +1 844... (11 digits) matches PDF that shows 844... (10 digits).
    Returns dict: "missing_in_pdf" = list of phone numbers not found in PDF; "all_found" = bool.
    """
    html_phones = list(dict.fromkeys(extract_phone_numbers_from_html(html_content)))  # dedupe, preserve order
    if not html_phones:
        return {"missing_in_pdf": [], "all_found": True}
    pdf_norm = _normalize_phone(pdf_text)
    missing = []
    for p in html_phones:
        if not _phone_found_in_text(p, pdf_norm):
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


