"""
Streamlit UI for PDF Comparison

A simple web interface to upload two PDFs and view differences with highlighting.
"""

import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
from compare_pdfs import PDFComparator
import io

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

def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory."""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

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

# Main UI
st.markdown('<h1 class="main-header">📄 PDF Comparison Tool</h1>', unsafe_allow_html=True)
st.markdown("### Upload two PDF files to compare and see differences highlighted in PDF2")

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
    pdf2_file = st.file_uploader(
        "Upload second PDF",
        type=['pdf'],
        key='pdf2',
        help="Upload the second PDF document. Differences will be highlighted in this PDF."
    )

# Compare button
if st.button("🔍 Compare PDFs", type="primary"):
    if pdf1_file is None or pdf2_file is None:
        st.error("⚠️ Please upload both PDF files to compare.")
    else:
        with st.spinner("Processing PDFs... This may take a moment."):
            try:
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    pdf1_path = save_uploaded_file(pdf1_file, temp_dir)
                    pdf2_path = save_uploaded_file(pdf2_file, temp_dir)
                    
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
                    
                    # Generate report
                    report = st.session_state.comparator.generate_comparison_report(
                        text1,
                        text2,
                        pdf1_file.name,
                        pdf2_file.name,
                        semantic_sim_max,
                        semantic_sim_avg,
                        text_diff
                    )
                    
                    # Store results in session state
                    # Note: We need to save PDFs to a persistent location for highlighting
                    temp_save_dir = Path(tempfile.gettempdir()) / "pdf_comparison"
                    temp_save_dir.mkdir(exist_ok=True)
                    
                    saved_pdf1 = temp_save_dir / f"pdf1_{pdf1_file.name}"
                    saved_pdf2 = temp_save_dir / f"pdf2_{pdf2_file.name}"
                    shutil.copy(pdf1_path, saved_pdf1)
                    shutil.copy(pdf2_path, saved_pdf2)
                    
                    st.session_state.report = report
                    st.session_state.text_diff = text_diff
                    st.session_state.pdf1_path = str(saved_pdf1)
                    st.session_state.pdf2_path = str(saved_pdf2)
                    st.session_state.comparison_done = True
                    
                    st.success("✅ Comparison complete!")
                    
            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")
                st.session_state.comparison_done = False

# Display results
if st.session_state.comparison_done and st.session_state.report:
    st.markdown("---")
    st.markdown("## 📊 Comparison Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Report", "🔍 Line Differences", "📄 Highlighted PDF"])
    
    with tab1:
        st.markdown("### Full Comparison Report")
        st.text_area(
            "Report",
            value=st.session_state.report,
            height=400,
            disabled=True
        )
        
        # Download report button
        st.download_button(
            label="📥 Download Report",
            data=st.session_state.report,
            file_name="comparison_report.txt",
            mime="text/plain"
        )
    
    with tab2:
        st.markdown("### Detailed Line-by-Line Differences")
        
        if st.session_state.text_diff and 'line_differences' in st.session_state.text_diff:
            line_diffs = st.session_state.text_diff['line_differences']
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Added Lines", st.session_state.text_diff.get('added_lines', 0))
            with col2:
                st.metric("Removed Lines", st.session_state.text_diff.get('removed_lines', 0))
            with col3:
                st.metric("Changed Lines", st.session_state.text_diff.get('changed_lines', 0))
            
            # Filter options
            st.markdown("#### Filter Differences")
            filter_type = st.selectbox(
                "Show:",
                ["All", "Added", "Removed", "Changed"],
                key="filter_type"
            )
            
            # Display filtered differences
            filtered_diffs = line_diffs
            if filter_type != "All":
                filtered_diffs = [d for d in line_diffs if d['type'] == filter_type.lower()]
            
            # Limit display
            max_display = st.slider("Maximum lines to display", 10, 100, 50)
            display_diffs = filtered_diffs[:max_display]
            
            st.markdown(f"#### Showing {len(display_diffs)} of {len(filtered_diffs)} differences")
            
            # Display differences
            for idx, diff in enumerate(display_diffs):
                with st.expander(f"Difference #{idx + 1} - {diff['type'].upper()}"):
                    if diff['type'] == 'added':
                        st.markdown(f"**Added at Line {diff['doc2_line']} in Document 2:**")
                        st.code(diff['content'], language=None)
                    elif diff['type'] == 'removed':
                        st.markdown(f"**Removed from Line {diff['doc1_line']} in Document 1:**")
                        st.code(diff['content'], language=None)
                    elif diff['type'] == 'changed':
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Document 1 (Line {diff['doc1_line']}):**")
                            st.code(diff['old_content'], language=None)
                        with col_b:
                            st.markdown(f"**Document 2 (Line {diff['doc2_line']}):**")
                            st.code(diff['new_content'], language=None)
        else:
            st.info("No line differences found.")
    
    with tab3:
        st.markdown("### Highlighted PDF (PDF2 with Differences)")
        st.markdown("**Legend:**")
        st.markdown("- 🟡 **Yellow highlight**: Added content")
        st.markdown("- 🟠 **Orange highlight**: Changed content")
        
        if st.button("🎨 Generate Highlighted PDF"):
            with st.spinner("Creating highlighted PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    output_path = tmp_file.name
                    
                    if highlight_pdf_differences(
                        st.session_state.pdf2_path,
                        st.session_state.text_diff,
                        output_path
                    ):
                        # Read the highlighted PDF
                        with open(output_path, 'rb') as f:
                            pdf_bytes = f.read()
                        
                        st.success("✅ Highlighted PDF created!")
                        
                        # Display download button
                        st.download_button(
                            label="📥 Download Highlighted PDF",
                            data=pdf_bytes,
                            file_name="pdf2_highlighted.pdf",
                            mime="application/pdf"
                        )
                        
                        # Display PDF preview
                        st.markdown("#### Preview:")
                        st.write("Note: Preview may not show highlights. Download the PDF to see highlights.")
                        import base64
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        
                        # Clean up
                        os.unlink(output_path)
                    else:
                        st.error("Failed to create highlighted PDF.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Hugging Face Models • No API keys required"
    "</div>",
    unsafe_allow_html=True
)

