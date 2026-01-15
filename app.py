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
                        pdf2_file.name,
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
                    saved_pdf2 = temp_save_dir / f"pdf2_{pdf2_file.name}"
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

# Display results
if st.session_state.comparison_done and st.session_state.report:
    st.markdown("---")
    st.markdown("## 📊 Comparison Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Report", "🔍 Line Differences", "🖼️ Images & Fonts", "📄 Visual PDF Comparison", "📄 Highlighted PDF"])
    
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
        st.markdown("### 🖼️ Image & Font Comparison")
        
        # Image Comparison Section
        if st.session_state.image_comparison:
            st.markdown("#### 📸 Image Comparison")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Images (Doc1)", st.session_state.image_comparison.get('total_images_1', 0))
            with col2:
                st.metric("Total Images (Doc2)", st.session_state.image_comparison.get('total_images_2', 0))
            with col3:
                st.metric("Similar Images", len(st.session_state.image_comparison.get('similar_images', [])))
            with col4:
                similarity_pct = st.session_state.image_comparison.get('similarity_score', 0) * 100
                st.metric("Image Similarity", f"{similarity_pct:.1f}%")
            with col5:
                spacing_stats = st.session_state.image_comparison.get('spacing_stats', {})
                same_spacing = spacing_stats.get('same_spacing', 0)
                total_compared = spacing_stats.get('total_compared', 0)
                if total_compared > 0:
                    spacing_pct = (same_spacing / total_compared) * 100
                    st.metric("Same Spacing", f"{same_spacing}/{total_compared} ({spacing_pct:.0f}%)")
                else:
                    st.metric("Same Spacing", "N/A")
            
            similar_images = st.session_state.image_comparison.get('similar_images', [])
            unique_to_1 = st.session_state.image_comparison.get('unique_to_1', 0)
            unique_to_2 = st.session_state.image_comparison.get('unique_to_2', 0)
            
            # Spacing Analysis Summary
            spacing_stats = st.session_state.image_comparison.get('spacing_stats', {})
            if spacing_stats.get('total_compared', 0) > 0:
                same_spacing = spacing_stats.get('same_spacing', 0)
                different_spacing = spacing_stats.get('different_spacing', 0)
                avg_similarity = spacing_stats.get('avg_spacing_similarity', 0) * 100
                
                if same_spacing > different_spacing:
                    st.success(f"✅ **Spacing Analysis**: {same_spacing} images have SAME spacing, {different_spacing} have DIFFERENT spacing (Avg similarity: {avg_similarity:.1f}%)")
                elif different_spacing > 0:
                    st.warning(f"⚠️ **Spacing Analysis**: {same_spacing} images have SAME spacing, {different_spacing} have DIFFERENT spacing (Avg similarity: {avg_similarity:.1f}%)")
            
            if unique_to_1 > 0 or unique_to_2 > 0:
                st.info(f"📊 {unique_to_1} images unique to Document 1, {unique_to_2} images unique to Document 2")
            
            if similar_images:
                st.markdown("##### Similar Image Pairs")
                max_display = st.slider("Maximum image pairs to display", 1, min(20, len(similar_images)), 5, key="img_display")
                
                for idx, sim_img in enumerate(similar_images[:max_display], 1):
                    spacing_status = sim_img.get('spacing_status', 'UNKNOWN')
                    spacing_sim = sim_img.get('spacing_similarity', 0) * 100
                    
                    # Color code based on spacing status
                    if spacing_status == 'SAME':
                        status_emoji = "✅"
                        status_color = "green"
                    else:
                        status_emoji = "⚠️"
                        status_color = "orange"
                    
                    with st.expander(f"{status_emoji} Image Pair #{idx} - Similarity: {sim_img['similarity']*100:.1f}% | Spacing: {spacing_status} ({spacing_sim:.1f}%)"):
                        img1_info = sim_img['image1']
                        img2_info = sim_img['image2']
                        
                        # Spacing details
                        width_diff = abs(img1_info['width'] - img2_info['width'])
                        height_diff = abs(img1_info['height'] - img2_info['height'])
                        
                        st.markdown(f"**Spacing Analysis:**")
                        col_sp1, col_sp2, col_sp3 = st.columns(3)
                        with col_sp1:
                            st.metric("Width Difference", f"{width_diff}px", 
                                     delta=f"{img1_info['width']}px → {img2_info['width']}px")
                        with col_sp2:
                            st.metric("Height Difference", f"{height_diff}px",
                                     delta=f"{img1_info['height']}px → {img2_info['height']}px")
                        with col_sp3:
                            st.metric("Spacing Status", spacing_status,
                                     delta=f"{spacing_sim:.1f}% similar")
                        
                        st.markdown("---")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Document 1**")
                            st.markdown(f"- Page: {img1_info['page']}")
                            st.markdown(f"- Size: {img1_info['width']}x{img1_info['height']}px")
                            st.markdown(f"- File size: {img1_info['size']:,} bytes")
                            
                            # Try to display image if available
                            if st.session_state.images1:
                                for img in st.session_state.images1:
                                    if img['page'] == img1_info['page'] and img['index'] == img1_info['index']:
                                        try:
                                            st.image(img['pil_image'], caption=f"Page {img1_info['page']}", use_container_width=True)
                                        except:
                                            st.info("Image preview not available")
                                        break
                        
                        with col_b:
                            st.markdown(f"**Document 2**")
                            st.markdown(f"- Page: {img2_info['page']}")
                            st.markdown(f"- Size: {img2_info['width']}x{img2_info['height']}px")
                            st.markdown(f"- File size: {img2_info['size']:,} bytes")
                            
                            # Try to display image if available
                            if st.session_state.images2:
                                for img in st.session_state.images2:
                                    if img['page'] == img2_info['page'] and img['index'] == img2_info['index']:
                                        try:
                                            st.image(img['pil_image'], caption=f"Page {img2_info['page']}", use_container_width=True)
                                        except:
                                            st.info("Image preview not available")
                                        break
            else:
                st.info("No similar images found between the documents.")
        else:
            st.info("Image comparison not available. Ensure PyMuPDF and Pillow are installed.")
        
        st.markdown("---")
        
        # Font Comparison Section
        if st.session_state.font_comparison:
            st.markdown("#### 🔤 Font Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fonts (Doc1)", st.session_state.font_comparison.get('font_count_1', 0))
            with col2:
                st.metric("Fonts (Doc2)", st.session_state.font_comparison.get('font_count_2', 0))
            with col3:
                st.metric("Common Fonts", st.session_state.font_comparison.get('common_count', 0))
            with col4:
                font_sim = st.session_state.font_comparison.get('similarity_score', 0) * 100
                st.metric("Font Similarity", f"{font_sim:.1f}%")
            
            common_fonts = st.session_state.font_comparison.get('common_fonts', [])
            only_in_1 = st.session_state.font_comparison.get('only_in_1', [])
            only_in_2 = st.session_state.font_comparison.get('only_in_2', [])
            
            # Show common fonts with text samples
            common_font_samples = st.session_state.font_comparison.get('common_font_samples', {})
            if common_fonts:
                st.markdown("##### ✅ Common Fonts (Used in Both Documents)")
                st.markdown("*Click to see text samples rendered as images from the PDFs*")
                
                for font_name in common_fonts[:10]:  # Show first 10 common fonts
                    with st.expander(f"🔤 **{font_name}** - Text Samples & Visual Comparison"):
                        samples_info = common_font_samples.get(font_name, {})
                        doc1_samples = samples_info.get('doc1_samples', [])
                        doc2_samples = samples_info.get('doc2_samples', [])
                        
                        if doc1_samples or doc2_samples:
                            # Try to get visual text block images
                            try:
                                if st.session_state.pdf1_path and st.session_state.pdf2_path:
                                    doc1_text_blocks = st.session_state.comparator.extract_text_blocks_with_fonts(
                                        st.session_state.pdf1_path, font_name, max_samples=3
                                    )
                                    doc2_text_blocks = st.session_state.comparator.extract_text_blocks_with_fonts(
                                        st.session_state.pdf2_path, font_name, max_samples=3
                                    )
                                    
                                    if doc1_text_blocks or doc2_text_blocks:
                                        st.markdown("**Visual Text Samples (Rendered from PDF):**")
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            st.markdown("**Document 1:**")
                                            for idx, block in enumerate(doc1_text_blocks[:3]):
                                                st.markdown(f"*Sample {idx+1} - Page {block['page']}, {block['size']}pt:*")
                                                st.image(block['image'], use_container_width=True, caption=block['text'][:50])
                                        
                                        with col_b:
                                            st.markdown("**Document 2:**")
                                            for idx, block in enumerate(doc2_text_blocks[:3]):
                                                st.markdown(f"*Sample {idx+1} - Page {block['page']}, {block['size']}pt:*")
                                                st.image(block['image'], use_container_width=True, caption=block['text'][:50])
                                        
                                        st.markdown("---")
                            except Exception as e:
                                st.warning(f"Could not render visual text samples: {str(e)}")
                            
                            # Show text content
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Document 1 Text Samples:**")
                                for sample in doc1_samples[:3]:  # Show first 3 samples
                                    text = sample.get('text', '')
                                    size = sample.get('size', 12)
                                    page = sample.get('page', 0)
                                    st.markdown(f"*Page {page}, Size {size}pt:*")
                                    st.code(text[:100] + ("..." if len(text) > 100 else ""), language=None)
                            
                            with col_b:
                                st.markdown("**Document 2 Text Samples:**")
                                for sample in doc2_samples[:3]:  # Show first 3 samples
                                    text = sample.get('text', '')
                                    size = sample.get('size', 12)
                                    page = sample.get('page', 0)
                                    st.markdown(f"*Page {page}, Size {size}pt:*")
                                    st.code(text[:100] + ("..." if len(text) > 100 else ""), language=None)
                        else:
                            st.info(f"Font '{font_name}' found but no text samples available.")
            
            if only_in_1:
                st.markdown("##### 📄 Fonts Only in Document 1")
                font_cols = st.columns(min(3, len(only_in_1)))
                for idx, font in enumerate(only_in_1):
                    with font_cols[idx % 3]:
                        st.markdown(f"• {font}")
            
            if only_in_2:
                st.markdown("##### 📄 Fonts Only in Document 2")
                font_cols = st.columns(min(3, len(only_in_2)))
                for idx, font in enumerate(only_in_2):
                    with font_cols[idx % 3]:
                        st.markdown(f"• {font}")
        else:
            st.info("Font comparison not available. Ensure PyMuPDF is installed.")
    
    with tab4:
        st.markdown("### 📄 Visual PDF Comparison")
        st.markdown("**Side-by-side page comparison to observe spacing, image placement, and layout changes**")
        
        if st.session_state.pdf_pages1 and st.session_state.pdf_pages2:
            # Page selector
            max_pages = max(len(st.session_state.pdf_pages1), len(st.session_state.pdf_pages2))
            if max_pages > 0:
                # Navigation buttons
                col_prev, col_info, col_next = st.columns([1, 2, 1])
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
                
                # Get page images
                page1_img = None
                page2_img = None
                
                for page_data in st.session_state.pdf_pages1:
                    if page_data['page_num'] == page_to_view:
                        page1_img = page_data['image']
                        break
                
                for page_data in st.session_state.pdf_pages2:
                    if page_data['page_num'] == page_to_view:
                        page2_img = page_data['image']
                        break
                
                if page1_img or page2_img:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### Document 1 - Page {page_to_view}")
                        if page1_img:
                            st.image(page1_img, use_container_width=True, caption=f"Page {page_to_view} from Document 1")
                        else:
                            st.info("Page not available in Document 1")
                    
                    with col2:
                        st.markdown(f"### Document 2 - Page {page_to_view}")
                        if page2_img:
                            st.image(page2_img, use_container_width=True, caption=f"Page {page_to_view} from Document 2")
                        else:
                            st.info("Page not available in Document 2")
                else:
                    st.warning(f"Page {page_to_view} not found in either document.")
            else:
                st.info("No pages available for comparison.")
        else:
            st.info("PDF pages not rendered. Please run the comparison again.")
    
    with tab5:
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

