"""
PDF Comparison Script using Hugging Face Models

Goal: Compare two PDF files using free Hugging Face models by extracting text,
generating embeddings for semantic similarity, and analyzing differences.

Inputs:
    - pdf1_path: Path to first PDF file
    - pdf2_path: Path to second PDF file

Outputs:
    - Comparison report with differences, similarities, and analysis

Constraints:
    - PDF files must exist and be readable
    - Models are downloaded on first use (requires internet connection)
    - Large PDFs may be truncated for processing
    - Processing happens locally (no API costs)

Edge Cases:
    - Missing files
    - Invalid PDF format
    - Network errors during model download
    - Large files requiring chunking (automatically handled)
    - Empty or corrupted PDFs
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import difflib

try:
    import PyPDF2
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: sentence-transformers and scikit-learn required.")
    print("Install with: pip install sentence-transformers scikit-learn")


class PDFComparator:
    """Handles PDF comparison using Hugging Face models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PDF comparator.
        
        Args:
            model_name: Hugging Face model name for embeddings.
                       Default: "all-MiniLM-L6-v2" (fast, lightweight)
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install sentence-transformers scikit-learn"
            )
        
        print(f"Loading Hugging Face model: {model_name}...")
        print("(This may take a moment on first use as the model downloads)")
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def validate_pdf(self, file_path: str) -> Path:
        """
        Validate that the PDF file exists and is readable.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Path object if valid
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        if path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {file_path}")
        return path
    
    def extract_text_from_pdf(self, file_path: str, max_chars: int = 500000) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            max_chars: Maximum characters to extract
            
        Returns:
            Extracted text
        """
        if not PDF_EXTRACTION_AVAILABLE:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        path = self.validate_pdf(file_path)
        text = ""
        
        try:
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    if len(text) >= max_chars:
                        text += f"\n\n[Content truncated at page {page_num + 1}...]"
                        break
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            raise Exception(f"Failed to extract text from {file_path}: {str(e)}")
        
        return text[:max_chars]
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better comparison.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Split into chunks for better comparison
        chunks1 = self.split_into_chunks(text1)
        chunks2 = self.split_into_chunks(text2)
        
        # Generate embeddings for all chunks
        embeddings1 = self.model.encode(chunks1, show_progress_bar=False)
        embeddings2 = self.model.encode(chunks2, show_progress_bar=False)
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        # Return maximum similarity (best matching chunks)
        max_similarity = float(np.max(similarity_matrix))
        
        # Also calculate average similarity for overall comparison
        avg_similarity = float(np.mean(similarity_matrix))
        
        return max_similarity, avg_similarity
    
    def find_text_differences(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Find textual differences between two texts with line numbers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with difference statistics and detailed line differences
        """
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        # Use SequenceMatcher to get detailed differences with line numbers
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        
        # Track line-by-line differences
        line_differences = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Lines match, skip
                continue
            elif tag == 'delete':
                # Lines removed from doc1
                for line_idx in range(i1, i2):
                    line_differences.append({
                        'type': 'removed',
                        'doc1_line': line_idx + 1,
                        'doc2_line': None,
                        'content': lines1[line_idx]
                    })
            elif tag == 'insert':
                # Lines added to doc2
                for line_idx in range(j1, j2):
                    line_differences.append({
                        'type': 'added',
                        'doc1_line': None,
                        'doc2_line': line_idx + 1,
                        'content': lines2[line_idx]
                    })
            elif tag == 'replace':
                # Lines changed
                max_len = max(i2 - i1, j2 - j1)
                for idx in range(max_len):
                    doc1_line = i1 + idx if idx < (i2 - i1) else None
                    doc2_line = j1 + idx if idx < (j2 - j1) else None
                    
                    if doc1_line is not None and doc2_line is not None:
                        # Both lines exist - it's a change
                        line_differences.append({
                            'type': 'changed',
                            'doc1_line': doc1_line + 1,
                            'doc2_line': doc2_line + 1,
                            'old_content': lines1[doc1_line],
                            'new_content': lines2[doc2_line]
                        })
                    elif doc1_line is not None:
                        # Line removed
                        line_differences.append({
                            'type': 'removed',
                            'doc1_line': doc1_line + 1,
                            'doc2_line': None,
                            'content': lines1[doc1_line]
                        })
                    elif doc2_line is not None:
                        # Line added
                        line_differences.append({
                            'type': 'added',
                            'doc1_line': None,
                            'doc2_line': doc2_line + 1,
                            'content': lines2[doc2_line]
                        })
        
        # Count statistics
        added_count = sum(1 for d in line_differences if d['type'] == 'added')
        removed_count = sum(1 for d in line_differences if d['type'] == 'removed')
        changed_count = sum(1 for d in line_differences if d['type'] == 'changed')
        
        # Word-level comparison
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        common_words = words1.intersection(words2)
        unique_to_1 = words1 - words2
        unique_to_2 = words2 - words1
        
        return {
            'added_lines': added_count,
            'removed_lines': removed_count,
            'changed_lines': changed_count,
            'line_differences': line_differences,
            'common_words': len(common_words),
            'unique_to_1': len(unique_to_1),
            'unique_to_2': len(unique_to_2),
            'word_overlap': len(common_words) / max(len(words1), len(words2), 1) * 100
        }
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from text (headings, paragraphs).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sections
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = text.splitlines()
        for line in lines:
            # Detect headings (lines in all caps, or lines with specific patterns)
            if line.strip() and (
                line.isupper() and len(line) > 3 and len(line) < 100
                or re.match(r'^#{1,6}\s+', line)  # Markdown headers
                or re.match(r'^\d+\.\s+[A-Z]', line)  # Numbered sections
            ):
                if current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = line.strip()[:100]  # Limit section name length
                current_content = []
            else:
                if line.strip():
                    current_content.append(line.strip())
        
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections
    
    def generate_comparison_report(
        self, 
        text1: str, 
        text2: str, 
        pdf1_name: str, 
        pdf2_name: str,
        semantic_sim_max: float,
        semantic_sim_avg: float,
        text_diff: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            text1: Text from first PDF
            text2: Text from second PDF
            pdf1_name: Name of first PDF
            pdf2_name: Name of second PDF
            semantic_sim_max: Maximum semantic similarity score
            semantic_sim_avg: Average semantic similarity score
            text_diff: Text difference statistics
            
        Returns:
            Comparison report as string
        """
        report = []
        report.append("=" * 60)
        report.append("PDF COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"\nDocument 1: {pdf1_name}")
        report.append(f"Document 2: {pdf2_name}")
        report.append(f"\nDocument 1 length: {len(text1):,} characters")
        report.append(f"Document 2 length: {len(text2):,} characters")
        report.append("\n" + "=" * 60)
        
        # Summary
        report.append("\n## SUMMARY")
        report.append("-" * 60)
        similarity_percent = semantic_sim_avg * 100
        if similarity_percent >= 90:
            similarity_desc = "very similar"
        elif similarity_percent >= 70:
            similarity_desc = "moderately similar"
        elif similarity_percent >= 50:
            similarity_desc = "somewhat different"
        else:
            similarity_desc = "very different"
        
        report.append(f"\nThe documents are {similarity_desc}.")
        report.append(f"Semantic Similarity Score: {similarity_percent:.2f}%")
        report.append(f"Maximum Section Similarity: {semantic_sim_max * 100:.2f}%")
        
        # Similarities
        report.append("\n## SIMILARITIES")
        report.append("-" * 60)
        report.append(f"• Common words: {text_diff['common_words']:,}")
        report.append(f"• Word overlap: {text_diff['word_overlap']:.2f}%")
        
        if text_diff['word_overlap'] > 50:
            report.append("\nThe documents share significant vocabulary, suggesting")
            report.append("they cover similar topics or are related documents.")
        
        # Differences
        report.append("\n## KEY DIFFERENCES")
        report.append("-" * 60)
        report.append(f"• Lines added in Document 2: {text_diff['added_lines']:,}")
        report.append(f"• Lines removed from Document 1: {text_diff['removed_lines']:,}")
        if 'changed_lines' in text_diff:
            report.append(f"• Lines changed: {text_diff['changed_lines']:,}")
        report.append(f"• Words unique to Document 1: {text_diff['unique_to_1']:,}")
        report.append(f"• Words unique to Document 2: {text_diff['unique_to_2']:,}")
        
        # Detailed Line-by-Line Differences
        if 'line_differences' in text_diff and text_diff['line_differences']:
            report.append("\n## LINE-BY-LINE DIFFERENCES")
            report.append("-" * 60)
            
            # Limit the number of differences shown to avoid overwhelming output
            max_differences = 100
            line_diffs = text_diff['line_differences'][:max_differences]
            
            if len(text_diff['line_differences']) > max_differences:
                report.append(f"\n(Showing first {max_differences} of {len(text_diff['line_differences'])} differences)")
                report.append("(Use the full report file for complete details)\n")
            
            # Group differences by type for better readability
            added_lines = [d for d in line_diffs if d['type'] == 'added']
            removed_lines = [d for d in line_diffs if d['type'] == 'removed']
            changed_lines = [d for d in line_diffs if d['type'] == 'changed']
            
            # Show removed lines
            if removed_lines:
                report.append(f"\n### Removed Lines (from {pdf1_name}):")
                for diff in removed_lines[:30]:  # Limit to first 30
                    line_num = diff['doc1_line']
                    content = diff['content']
                    # Truncate very long lines
                    if len(content) > 100:
                        content = content[:97] + "..."
                    report.append(f"  Line {line_num}: {content}")
                if len(removed_lines) > 30:
                    report.append(f"  ... and {len(removed_lines) - 30} more removed lines")
            
            # Show added lines
            if added_lines:
                report.append(f"\n### Added Lines (in {pdf2_name}):")
                for diff in added_lines[:30]:  # Limit to first 30
                    line_num = diff['doc2_line']
                    content = diff['content']
                    # Truncate very long lines
                    if len(content) > 100:
                        content = content[:97] + "..."
                    report.append(f"  Line {line_num}: {content}")
                if len(added_lines) > 30:
                    report.append(f"  ... and {len(added_lines) - 30} more added lines")
            
            # Show changed lines
            if changed_lines:
                report.append(f"\n### Changed Lines:")
                for diff in changed_lines[:20]:  # Limit to first 20
                    doc1_line = diff['doc1_line']
                    doc2_line = diff['doc2_line']
                    old_content = diff['old_content']
                    new_content = diff['new_content']
                    
                    # Truncate very long lines
                    if len(old_content) > 80:
                        old_content = old_content[:77] + "..."
                    if len(new_content) > 80:
                        new_content = new_content[:77] + "..."
                    
                    report.append(f"  Doc1 Line {doc1_line} → Doc2 Line {doc2_line}:")
                    report.append(f"    - {old_content}")
                    report.append(f"    + {new_content}")
                if len(changed_lines) > 20:
                    report.append(f"  ... and {len(changed_lines) - 20} more changed lines")
            
            # Summary of all differences
            report.append(f"\n### Difference Summary:")
            report.append(f"  Total differences found: {len(text_diff['line_differences']):,}")
            report.append(f"  - Removed: {len(removed_lines):,}")
            report.append(f"  - Added: {len(added_lines):,}")
            report.append(f"  - Changed: {len(changed_lines):,}")
        
        # Detailed Analysis
        report.append("\n## DETAILED ANALYSIS")
        report.append("-" * 60)
        
        # Extract and compare sections
        sections1 = self.extract_key_sections(text1)
        sections2 = self.extract_key_sections(text2)
        
        if sections1 or sections2:
            report.append("\n### Section Comparison:")
            all_sections = set(list(sections1.keys()) + list(sections2.keys()))
            
            for section in sorted(all_sections)[:20]:  # Limit to first 20 sections
                in_doc1 = section in sections1
                in_doc2 = section in sections2
                
                if in_doc1 and in_doc2:
                    # Compare section content
                    content1 = sections1[section]
                    content2 = sections2[section]
                    
                    if len(content1) > 500 or len(content2) > 500:
                        # For long sections, calculate similarity
                        sim_max, sim_avg = self.calculate_semantic_similarity(content1[:5000], content2[:5000])
                        report.append(f"\n✓ Section '{section[:50]}...' appears in both documents")
                        report.append(f"  Section similarity: {sim_avg * 100:.1f}%")
                    else:
                        report.append(f"\n✓ Section '{section[:50]}...' appears in both documents")
                elif in_doc1:
                    report.append(f"\n⚠ Section '{section[:50]}...' only in Document 1")
                elif in_doc2:
                    report.append(f"\n⚠ Section '{section[:50]}...' only in Document 2")
        
        # Conclusion
        report.append("\n## CONCLUSION")
        report.append("-" * 60)
        
        if similarity_percent >= 90:
            report.append("\nThese documents are highly similar. They likely represent")
            report.append("different versions of the same document or cover the same")
            report.append("content with minor variations.")
        elif similarity_percent >= 70:
            report.append("\nThese documents share significant content but have notable")
            report.append("differences. They may be related documents or different")
            report.append("versions with substantial changes.")
        elif similarity_percent >= 50:
            report.append("\nThese documents have moderate similarity. They may cover")
            report.append("related topics but with different focus or content.")
        else:
            report.append("\nThese documents are quite different. They likely cover")
            report.append("different topics or have substantially different content.")
        
        report.append(f"\nOverall similarity: {similarity_percent:.2f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def compare_pdfs(self, pdf1_path: str, pdf2_path: str) -> str:
        """
        Compare two PDF files using Hugging Face models.
        
        Args:
            pdf1_path: Path to first PDF
            pdf2_path: Path to second PDF
            
        Returns:
            Comparison report as string
        """
        print("\n" + "="*60)
        print("PDF Comparison Process (Hugging Face Models)")
        print("="*60)
        print("Using free Hugging Face models - no API key required!\n")
        
        if not PDF_EXTRACTION_AVAILABLE:
            raise ImportError(
                "PyPDF2 is required for text extraction. "
                "Install with: pip install PyPDF2"
            )
        
        # Extract text from both PDFs
        print(f"Extracting text from {Path(pdf1_path).name}...")
        text1 = self.extract_text_from_pdf(pdf1_path)
        print(f"✓ Extracted {len(text1):,} characters from {Path(pdf1_path).name}")
        
        print(f"Extracting text from {Path(pdf2_path).name}...")
        text2 = self.extract_text_from_pdf(pdf2_path)
        print(f"✓ Extracted {len(text2):,} characters from {Path(pdf2_path).name}")
        
        # Calculate semantic similarity
        print("\nCalculating semantic similarity...")
        semantic_sim_max, semantic_sim_avg = self.calculate_semantic_similarity(text1, text2)
        print(f"✓ Semantic similarity: {semantic_sim_avg * 100:.2f}%")
        
        # Find text differences
        print("Analyzing text differences...")
        text_diff = self.find_text_differences(text1, text2)
        print("✓ Text analysis complete")
        
        # Generate comparison report
        print("\nGenerating comparison report...")
        report = self.generate_comparison_report(
            text1, 
            text2, 
            Path(pdf1_path).name, 
            Path(pdf2_path).name,
            semantic_sim_max,
            semantic_sim_avg,
            text_diff
        )
        print("✓ Report generated\n")
        
        return report


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 3:
        print("Usage: python compare_pdfs.py <pdf1_path> <pdf2_path>")
        print("\nExample:")
        print("  python compare_pdfs.py document1.pdf document2.pdf")
        print("\nNote: This script uses free Hugging Face models - no API key required!")
        sys.exit(1)
    
    pdf1_path = sys.argv[1]
    pdf2_path = sys.argv[2]
    
    try:
        # Initialize comparator
        comparator = PDFComparator()
        
        # Perform comparison
        result = comparator.compare_pdfs(pdf1_path, pdf2_path)
        
        # Display results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(result)
        print("="*60)
        
        # Save to file
        output_file = "comparison_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n✓ Results saved to {output_file}")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}", file=sys.stderr)
        print("\nPlease install required packages:", file=sys.stderr)
        print("  pip install sentence-transformers scikit-learn PyPDF2", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
