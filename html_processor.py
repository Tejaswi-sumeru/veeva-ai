import re
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
