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
from typing import List, Dict, Any, Tuple, Optional
import difflib
import io
import hashlib

# Placeholder for dynamic content (bracketed alternatives). Stripped before comparison so we don't compare it.
DYNAMIC_PLACEHOLDER = "{{DYNAMIC}}"

# Match bracketed content that contains a slash (e.g. [EXPLORE / NEW IN] or [A / B / C]). Structure-based, no variable names.
_BRACKETED_ALTERNATIVES_PATTERN = re.compile(r"\[[^\]]*\/[^\]]*\]")


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison: replace bracketed alternatives [X / Y] with a placeholder,
    then remove the placeholder so we don't compare dynamic content. Returns text suitable
    for semantic similarity and diff (no {{DYNAMIC}} in output).
    """
    if not text:
        return text
    # Replace [ ... / ... ] with placeholder (structure-based, no literal variable names)
    normalized = _BRACKETED_ALTERNATIVES_PATTERN.sub(DYNAMIC_PLACEHOLDER, text)
    # Remove placeholder so we don't compare it; use space to avoid gluing words
    normalized = normalized.replace(DYNAMIC_PLACEHOLDER, " ")
    # Collapse multiple spaces/newlines for cleaner comparison
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
    return normalized.strip()


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

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Image and font extraction will be limited.")

try:
    from PIL import Image
    import imagehash
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    print("Warning: Pillow and imagehash not installed. Image comparison will be limited.")


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
    
    def extract_images_from_pdf(self, file_path: str, max_images: int = 50) -> List[Dict[str, Any]]:
        """
        Extract images from PDF file using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            max_images: Maximum number of images to extract
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        if not PYMUPDF_AVAILABLE:
            return []
        
        if not IMAGE_PROCESSING_AVAILABLE:
            return []
        
        path = self.validate_pdf(file_path)
        images = []
        
        try:
            doc = fitz.open(path)
            image_count = 0
            
            for page_num in range(len(doc)):
                if image_count >= max_images:
                    break
                    
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    if image_count >= max_images:
                        break
                    
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Create PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Calculate image hash for comparison
                        img_hash = imagehash.phash(pil_image)
                        
                        # Get image dimensions
                        width, height = pil_image.size
                        
                        # Calculate image size
                        image_size = len(image_bytes)
                        
                        images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'xref': xref,
                            'width': width,
                            'height': height,
                            'size': image_size,
                            'format': image_ext,
                            'hash': str(img_hash),
                            'image_bytes': image_bytes,
                            'pil_image': pil_image
                        })
                        
                        image_count += 1
                    except Exception as e:
                        # Skip images that can't be processed
                        continue
            
            doc.close()
        except Exception as e:
            print(f"Warning: Failed to extract images from {file_path}: {str(e)}")
        
        return images
    
    def extract_fonts_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract font information from PDF file using PyMuPDF.
        Also extracts text blocks with font information for visual comparison.
        Uses multiple methods to ensure font detection works.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing font information and text samples
        """
        if not PYMUPDF_AVAILABLE:
            return {'fonts': [], 'font_count': 0, 'unique_fonts': set(), 'font_text_samples': {}}
        
        path = self.validate_pdf(file_path)
        fonts_info = []
        unique_fonts = set()
        font_text_samples = {}  # font_name -> list of text samples with metadata
        
        try:
            doc = fitz.open(path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Method 1: Get fonts from page.get_fonts()
                font_list = page.get_fonts(full=True)
                
                # Method 2: Extract text blocks with font information (more reliable)
                text_dict = page.get_text("dict")
                
                # Process text spans to extract fonts (primary method)
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                font_name = span.get("font", "")
                                font_size = span.get("size", 12)
                                text_content = span.get("text", "").strip()
                                
                                if not font_name or font_name == "Unknown":
                                    continue
                                
                                # Clean font name - remove subset prefix
                                original_font = font_name
                                if '+' in font_name:
                                    font_name = font_name.split('+')[1]
                                
                                # Remove common prefixes/suffixes
                                font_name = font_name.replace('CIDFont+', '').replace('TrueType+', '')
                                font_name = re.sub(r'^[A-Z0-9]+[+-]', '', font_name)  # Remove encoded prefixes
                                
                                # Normalize font name
                                font_name_clean = font_name.strip()
                                if font_name_clean and font_name_clean.lower() != 'unknown':
                                    unique_fonts.add(font_name_clean.lower())
                                    
                                    # Store font info
                                    fonts_info.append({
                                        'page': page_num + 1,
                                        'name': font_name_clean,
                                        'type': span.get("flags", 0),
                                        'size': font_size,
                                        'original': original_font
                                    })
                                    
                                    # Store text samples
                                    if text_content and len(text_content) > 2:
                                        if font_name_clean.lower() not in font_text_samples:
                                            font_text_samples[font_name_clean.lower()] = []
                                        
                                        font_text_samples[font_name_clean.lower()].append({
                                            'text': text_content[:200],
                                            'size': font_size,
                                            'page': page_num + 1,
                                            'bbox': span.get("bbox", [0, 0, 0, 0])
                                        })
                
                # Method 3: Also check page.get_fonts() for additional fonts
                for font_info in font_list:
                    # Handle both dict and tuple formats from get_fonts()
                    if isinstance(font_info, dict):
                        font_name = font_info.get('name', '')
                        font_base = font_info.get('basefont', '')
                    elif isinstance(font_info, (tuple, list)) and len(font_info) >= 2:
                        # Tuple format: (xref, ext, type, basefont, name, encoding)
                        # Index: 0=xref, 1=ext, 2=type, 3=basefont, 4=name, 5=encoding
                        font_name = font_info[4] if len(font_info) > 4 else ''
                        font_base = font_info[3] if len(font_info) > 3 else ''
                    else:
                        continue
                    
                    # Use basefont if available, otherwise use name
                    font_to_use = font_base if font_base and font_base != 'Unknown' else font_name
                    
                    if not font_to_use or font_to_use == 'Unknown':
                        continue
                    
                    # Clean font name
                    if '+' in font_to_use:
                        font_to_use = font_to_use.split('+')[1]
                    
                    font_to_use = font_to_use.replace('CIDFont+', '').replace('TrueType+', '')
                    font_to_use = re.sub(r'^[A-Z0-9]+[+-]', '', font_to_use).strip()
                    
                    if font_to_use and font_to_use.lower() != 'unknown':
                        unique_fonts.add(font_to_use.lower())
            
            doc.close()
        except Exception as e:
            print(f"Warning: Failed to extract fonts from {file_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return {
            'fonts': fonts_info,
            'font_count': len(fonts_info),
            'unique_fonts': unique_fonts,
            'unique_count': len(unique_fonts),
            'font_text_samples': font_text_samples
        }
    
    def render_pdf_page_to_image(self, file_path: str, page_num: int, zoom: float = 2.0) -> Optional[Image.Image]:
        """
        Render a PDF page to a PIL Image.
        
        Args:
            file_path: Path to PDF file
            page_num: Page number (0-indexed)
            zoom: Zoom factor for rendering (higher = better quality)
            
        Returns:
            PIL Image or None if failed
        """
        if not PYMUPDF_AVAILABLE or not IMAGE_PROCESSING_AVAILABLE:
            return None
        
        try:
            doc = fitz.open(file_path)
            if page_num >= len(doc):
                doc.close()
                return None
            
            page = doc[page_num]
            w_pt, h_pt = page.rect.width, page.rect.height
            max_dim_px = 4096
            max_pt = max(w_pt, h_pt)
            if max_pt > 0 and max(w_pt * zoom, h_pt * zoom) > max_dim_px:
                zoom = min(zoom, max_dim_px / max_pt)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if img.width > max_dim_px or img.height > max_dim_px:
                ratio = max_dim_px / max(img.width, img.height)
                new_w = max(1, int(img.width * ratio))
                new_h = max(1, int(img.height * ratio))
                resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                img = img.resize((new_w, new_h), resample)
            doc.close()
            return img
        except Exception as e:
            print(f"Warning: Failed to render page {page_num} from {file_path}: {str(e)}")
            return None
    
    def render_all_pdf_pages(self, file_path: str, max_pages: int = 50, zoom: float = 2.0) -> List[Dict[str, Any]]:
        """
        Render all PDF pages to images.
        
        Args:
            file_path: Path to PDF file
            max_pages: Maximum number of pages to render
            zoom: Zoom factor for rendering
            
        Returns:
            List of dictionaries with page images and metadata
        """
        if not PYMUPDF_AVAILABLE or not IMAGE_PROCESSING_AVAILABLE:
            return []
        
        pages = []
        try:
            doc = fitz.open(file_path)
            num_pages = min(len(doc), max_pages)
            
            for page_num in range(num_pages):
                img = self.render_pdf_page_to_image(file_path, page_num, zoom)
                if img:
                    pages.append({
                        'page_num': page_num + 1,
                        'image': img,
                        'width': img.width,
                        'height': img.height
                    })
            
            doc.close()
        except Exception as e:
            print(f"Warning: Failed to render pages from {file_path}: {str(e)}")
        
        return pages

    def extract_text_blocks_with_fonts(self, file_path: str, font_name: str, max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Extract text blocks that use a specific font and render them as images from the PDF.
        
        Args:
            file_path: Path to PDF file
            font_name: Font name to extract
            max_samples: Maximum number of samples to extract
            
        Returns:
            List of dictionaries with text blocks and their rendered images
        """
        if not PYMUPDF_AVAILABLE or not IMAGE_PROCESSING_AVAILABLE:
            return []
        
        samples = []
        font_name_lower = font_name.lower()
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                if len(samples) >= max_samples:
                    break
                
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                # Render the page as image
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                for block in text_dict.get("blocks", []):
                    if len(samples) >= max_samples:
                        break
                    
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                span_font = span.get("font", "")
                                
                                # Clean font name
                                if '+' in span_font:
                                    span_font = span_font.split('+')[1]
                                
                                if span_font.lower() == font_name_lower:
                                    text_content = span.get("text", "").strip()
                                    bbox = span.get("bbox", [0, 0, 0, 0])
                                    
                                    if text_content and len(text_content) > 2:
                                        # Crop the text region from the page image
                                        # Scale bbox to match rendered image
                                        scale = 2.0
                                        x0, y0, x1, y1 = [int(coord * scale) for coord in bbox]
                                        
                                        # Ensure coordinates are within image bounds
                                        x0 = max(0, min(x0, page_img.width))
                                        y0 = max(0, min(y0, page_img.height))
                                        x1 = max(0, min(x1, page_img.width))
                                        y1 = max(0, min(y1, page_img.height))
                                        
                                        if x1 > x0 and y1 > y0:
                                            try:
                                                text_img = page_img.crop((x0, y0, x1, y1))
                                                
                                                samples.append({
                                                    'text': text_content[:200],
                                                    'page': page_num + 1,
                                                    'size': span.get("size", 12),
                                                    'image': text_img,
                                                    'bbox': bbox
                                                })
                                            except:
                                                pass
                                
                                if len(samples) >= max_samples:
                                    break
                            if len(samples) >= max_samples:
                                break
                    if len(samples) >= max_samples:
                        break
            
            doc.close()
        except Exception as e:
            print(f"Warning: Failed to extract text blocks for font {font_name}: {str(e)}")
        
        return samples
    
    def compare_images(self, images1: List[Dict[str, Any]], images2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare images from two PDFs, including spacing analysis.
        
        Args:
            images1: List of images from first PDF
            images2: List of images from second PDF
            
        Returns:
            Dictionary with comparison results including spacing analysis
        """
        if not IMAGE_PROCESSING_AVAILABLE:
            return {
                'similar_images': [],
                'unique_to_1': len(images1),
                'unique_to_2': len(images2),
                'similarity_score': 0.0,
                'spacing_analysis': []
            }
        
        similar_images = []
        matched_indices_2 = set()
        spacing_analysis = []
        
        # Compare images using perceptual hashing
        for img1 in images1:
            best_match = None
            best_similarity = 0.0
            best_index = -1
            
            hash1 = imagehash.hex_to_hash(img1['hash'])
            
            for idx, img2 in enumerate(images2):
                if idx in matched_indices_2:
                    continue
                
                hash2 = imagehash.hex_to_hash(img2['hash'])
                
                # Calculate hamming distance (lower = more similar)
                hamming_distance = hash1 - hash2
                
                # Convert to similarity score (0-1, where 1 is identical)
                # Max hamming distance for phash is 64
                similarity = 1.0 - (hamming_distance / 64.0)
                
                if similarity > best_similarity and similarity > 0.7:  # Threshold for similarity
                    best_similarity = similarity
                    best_match = img2
                    best_index = idx
            
            if best_match:
                # Analyze spacing differences
                width_diff = abs(img1['width'] - best_match['width'])
                height_diff = abs(img1['height'] - best_match['height'])
                size_diff = abs(img1['size'] - best_match['size'])
                
                # Calculate spacing similarity
                width_similarity = 1.0 - (width_diff / max(img1['width'], best_match['width'], 1))
                height_similarity = 1.0 - (height_diff / max(img1['height'], best_match['height'], 1))
                spacing_similarity = (width_similarity + height_similarity) / 2.0
                
                # Determine if spacing is same or different
                spacing_status = "SAME" if spacing_similarity > 0.95 else "DIFFERENT"
                
                spacing_analysis.append({
                    'page1': img1['page'],
                    'page2': best_match['page'],
                    'spacing_status': spacing_status,
                    'spacing_similarity': spacing_similarity,
                    'width_diff_px': width_diff,
                    'height_diff_px': height_diff,
                    'size_diff_bytes': size_diff,
                    'image1_size': f"{img1['width']}x{img1['height']}",
                    'image2_size': f"{best_match['width']}x{best_match['height']}"
                })
                
                similar_images.append({
                    'image1': {
                        'page': img1['page'],
                        'index': img1['index'],
                        'width': img1['width'],
                        'height': img1['height'],
                        'size': img1['size']
                    },
                    'image2': {
                        'page': best_match['page'],
                        'index': best_match['index'],
                        'width': best_match['width'],
                        'height': best_match['height'],
                        'size': best_match['size']
                    },
                    'similarity': best_similarity,
                    'spacing_status': spacing_status,
                    'spacing_similarity': spacing_similarity
                })
                matched_indices_2.add(best_index)
        
        unique_to_1 = len(images1) - len(similar_images)
        unique_to_2 = len(images2) - len(similar_images)
        
        # Calculate overall similarity score
        total_images = max(len(images1), len(images2), 1)
        similarity_score = len(similar_images) / total_images
        
        # Calculate spacing statistics
        same_spacing_count = sum(1 for s in spacing_analysis if s['spacing_status'] == 'SAME')
        different_spacing_count = len(spacing_analysis) - same_spacing_count
        avg_spacing_similarity = sum(s['spacing_similarity'] for s in spacing_analysis) / max(len(spacing_analysis), 1)
        
        return {
            'similar_images': similar_images,
            'unique_to_1': unique_to_1,
            'unique_to_2': unique_to_2,
            'similarity_score': similarity_score,
            'total_images_1': len(images1),
            'total_images_2': len(images2),
            'spacing_analysis': spacing_analysis,
            'spacing_stats': {
                'same_spacing': same_spacing_count,
                'different_spacing': different_spacing_count,
                'avg_spacing_similarity': avg_spacing_similarity,
                'total_compared': len(spacing_analysis)
            }
        }
    
    def compare_fonts(self, fonts1: Dict[str, Any], fonts2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare fonts from two PDFs.
        
        Args:
            fonts1: Font information from first PDF
            fonts2: Font information from second PDF
            
        Returns:
            Dictionary with font comparison results including text samples
        """
        unique_fonts_1 = fonts1.get('unique_fonts', set())
        unique_fonts_2 = fonts2.get('unique_fonts', set())
        
        common_fonts = unique_fonts_1.intersection(unique_fonts_2)
        only_in_1 = unique_fonts_1 - unique_fonts_2
        only_in_2 = unique_fonts_2 - unique_fonts_1
        
        # Calculate similarity score
        total_unique = len(unique_fonts_1.union(unique_fonts_2))
        similarity_score = len(common_fonts) / max(total_unique, 1)
        
        # Get text samples for common fonts
        font_text_samples_1 = fonts1.get('font_text_samples', {})
        font_text_samples_2 = fonts2.get('font_text_samples', {})
        
        common_font_samples = {}
        for font_name in common_fonts:
            samples_1 = font_text_samples_1.get(font_name, [])
            samples_2 = font_text_samples_2.get(font_name, [])
            if samples_1 or samples_2:
                common_font_samples[font_name] = {
                    'doc1_samples': samples_1[:5],  # Limit to 5 samples per font
                    'doc2_samples': samples_2[:5]
                }
        
        return {
            'common_fonts': sorted(list(common_fonts)),
            'only_in_1': sorted(list(only_in_1)),
            'only_in_2': sorted(list(only_in_2)),
            'similarity_score': similarity_score,
            'font_count_1': fonts1.get('unique_count', 0),
            'font_count_2': fonts2.get('unique_count', 0),
            'common_count': len(common_fonts),
            'common_font_samples': common_font_samples,
            'font_text_samples_1': font_text_samples_1,
            'font_text_samples_2': font_text_samples_2
        }
    
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
        Uses hybrid approach: text presence check + line-by-line comparison.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with difference statistics and detailed line differences
        """
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        # Build a set of normalized content from both documents for presence checking
        # This helps filter out content that exists in both but on different lines
        def normalize_for_presence(line: str) -> str:
            """Normalize line for presence checking: strip whitespace, lowercase"""
            return re.sub(r'\s+', ' ', line.strip().lower())
        
        # Create sets of normalized content for quick lookup
        content_in_doc1 = {normalize_for_presence(line) for line in lines1 if line.strip()}
        content_in_doc2 = {normalize_for_presence(line) for line in lines2 if line.strip()}
        
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
                    # Check if this content exists anywhere in doc2 (text presence check)
                    normalized_content = normalize_for_presence(lines1[line_idx])
                    if normalized_content and normalized_content in content_in_doc2:
                        # Content exists in doc2, just on a different line - skip highlighting
                        continue
                        
                    line_differences.append({
                        'type': 'removed',
                        'doc1_line': line_idx + 1,
                        'doc2_line': None,
                        'content': lines1[line_idx]
                    })
            elif tag == 'insert':
                # Lines added to doc2
                for line_idx in range(j1, j2):
                    # Check if this content exists anywhere in doc1 (text presence check)
                    normalized_content = normalize_for_presence(lines2[line_idx])
                    if normalized_content and normalized_content in content_in_doc1:
                        # Content exists in doc1, just on a different line - skip highlighting
                        continue
                        
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
                        # Both lines exist - check if they're actually different
                        # Use similarity ratio to avoid marking minor formatting differences as changes
                        line1_stripped = lines1[doc1_line].strip()
                        line2_stripped = lines2[doc2_line].strip()
                        
                        # Calculate similarity ratio
                        similarity_ratio = difflib.SequenceMatcher(None, line1_stripped, line2_stripped).ratio()
                        
                        # Only mark as changed if similarity is below threshold (0.85 = 85% similar)
                        # This filters out lines that are nearly identical but differ in whitespace/formatting
                        if similarity_ratio < 0.85:
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
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into sentences or meaningful chunks for comparison.
        Splits by punctuation and double newlines.
        """
        # Split by sentence endings, semicolons, and double newlines
        chunks = re.split(r'(?:[.!?]+\s+|\n\n+|;\s+)', text)
        
        # Filter out very short chunks (less than 3 characters)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 3]
    
    def _normalize_text_full(self, text: str) -> str:
        """
        Apply deep normalization to text to handle environment differences.
        """
        import unicodedata
        # Normalize unicode (NFKC is best for visual similarity)
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Collapse all whitespace to single spaces
        text = re.sub(r'[\s\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000\uFEFF]+', ' ', text)
        # Remove non-printable/control characters
        text = "".join(ch for ch in text if ch.isprintable())
        return text.strip()

    def _has_fuzzy_match(self, chunk: str, candidates: set, threshold: float = 0.97) -> bool:
        """
        Check if chunk has a fuzzy match in candidates.
        Threshold 0.97 is robust against minor line-ending/invisible char artifacts on servers.
        """
        for candidate in candidates:
            if chunk == candidate:
                return True
            # Ratio is 2*M/T where M is matches, T is total chars
            ratio = difflib.SequenceMatcher(None, chunk, candidate).ratio()
            if ratio >= threshold:
                return True
        return False
    
    def find_text_differences_chunk_based(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare texts using chunk-based presence checking.
        """
        # Split into chunks
        chunks1 = self._split_into_chunks(text1)
        chunks2 = self._split_into_chunks(text2)
        
        # Map normalized -> original
        normalized1 = {self._normalize_text_full(c): c for c in chunks1 if c.strip()}
        normalized2 = {self._normalize_text_full(c): c for c in chunks2 if c.strip()}
        
        # Find chunks only in doc1 (removed)
        only_in_1 = []
        for norm, original in normalized1.items():
            if norm not in normalized2:
                # Use robust threshold
                if not self._has_fuzzy_match(norm, normalized2.keys(), threshold=0.97):
                    only_in_1.append(original)
        
        # Find chunks only in doc2 (added)
        only_in_2 = []
        for norm, original in normalized2.items():
            if norm not in normalized1:
                if not self._has_fuzzy_match(norm, normalized1.keys(), threshold=0.97):
                    only_in_2.append(original)
        
        # Statistics
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        common_words = words1.intersection(words2)
        
        return {
            'added_lines': len(only_in_2),
            'removed_lines': len(only_in_1),
            'changed_lines': 0,
            'removed_chunks': only_in_1,
            'added_chunks': only_in_2,
            'common_words': len(common_words),
            'unique_to_1': len(words1 - words2),
            'unique_to_2': len(words2 - words1),
            'word_overlap': len(common_words) / max(len(words1), len(words2), 1) * 100,
            'comparison_mode': 'chunk_based',
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
        text_diff: Dict[str, Any],
        image_comparison: Optional[Dict[str, Any]] = None,
        font_comparison: Optional[Dict[str, Any]] = None
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
        
        # Image Comparison
        if image_comparison:
            report.append("\n## IMAGE COMPARISON")
            report.append("-" * 60)
            report.append(f"\nTotal images in Document 1: {image_comparison.get('total_images_1', 0)}")
            report.append(f"Total images in Document 2: {image_comparison.get('total_images_2', 0)}")
            report.append(f"Similar images found: {len(image_comparison.get('similar_images', []))}")
            report.append(f"Images unique to Document 1: {image_comparison.get('unique_to_1', 0)}")
            report.append(f"Images unique to Document 2: {image_comparison.get('unique_to_2', 0)}")
            report.append(f"Image similarity score: {image_comparison.get('similarity_score', 0) * 100:.2f}%")
            
            similar_images = image_comparison.get('similar_images', [])
            if similar_images:
                report.append("\n### Similar Images:")
                for idx, sim_img in enumerate(similar_images[:10], 1):  # Show first 10
                    img1_info = sim_img['image1']
                    img2_info = sim_img['image2']
                    similarity = sim_img['similarity']
                    report.append(f"\n  Image Pair #{idx}:")
                    report.append(f"    Doc1: Page {img1_info['page']}, {img1_info['width']}x{img1_info['height']}px")
                    report.append(f"    Doc2: Page {img2_info['page']}, {img2_info['width']}x{img2_info['height']}px")
                    report.append(f"    Similarity: {similarity * 100:.1f}%")
        
        # Font Comparison
        if font_comparison:
            report.append("\n## FONT COMPARISON")
            report.append("-" * 60)
            report.append(f"\nUnique fonts in Document 1: {font_comparison.get('font_count_1', 0)}")
            report.append(f"Unique fonts in Document 2: {font_comparison.get('font_count_2', 0)}")
            report.append(f"Common fonts: {font_comparison.get('common_count', 0)}")
            report.append(f"Fonts only in Document 1: {len(font_comparison.get('only_in_1', []))}")
            report.append(f"Fonts only in Document 2: {len(font_comparison.get('only_in_2', []))}")
            report.append(f"Font similarity score: {font_comparison.get('similarity_score', 0) * 100:.2f}%")
            
            common_fonts = font_comparison.get('common_fonts', [])
            if common_fonts:
                report.append("\n### Common Fonts:")
                for font in common_fonts[:20]:  # Show first 20
                    report.append(f"  • {font}")
            
            only_in_1 = font_comparison.get('only_in_1', [])
            if only_in_1:
                report.append("\n### Fonts Only in Document 1:")
                for font in only_in_1[:10]:  # Show first 10
                    report.append(f"  • {font}")
            
            only_in_2 = font_comparison.get('only_in_2', [])
            if only_in_2:
                report.append("\n### Fonts Only in Document 2:")
                for font in only_in_2[:10]:  # Show first 10
                    report.append(f"  • {font}")
        
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
        
        # Normalize: replace bracketed alternatives [X / Y] and strip {{DYNAMIC}} so we don't compare dynamic content
        text1 = normalize_for_comparison(text1)
        text2 = normalize_for_comparison(text2)
        
        # Calculate semantic similarity
        print("\nCalculating semantic similarity...")
        semantic_sim_max, semantic_sim_avg = self.calculate_semantic_similarity(text1, text2)
        print(f"✓ Semantic similarity: {semantic_sim_avg * 100:.2f}%")
        
        # Find text differences
        print("Analyzing text differences...")
        text_diff = self.find_text_differences(text1, text2)
        print("✓ Text analysis complete")
        
        # Extract and compare images
        image_comparison = None
        if PYMUPDF_AVAILABLE and IMAGE_PROCESSING_AVAILABLE:
            print("\nExtracting images from PDFs...")
            try:
                images1 = self.extract_images_from_pdf(pdf1_path)
                images2 = self.extract_images_from_pdf(pdf2_path)
                print(f"✓ Extracted {len(images1)} images from {Path(pdf1_path).name}")
                print(f"✓ Extracted {len(images2)} images from {Path(pdf2_path).name}")
                
                if images1 or images2:
                    print("Comparing images...")
                    image_comparison = self.compare_images(images1, images2)
                    print(f"✓ Image comparison complete: {len(image_comparison['similar_images'])} similar images found")
            except Exception as e:
                print(f"⚠ Warning: Image comparison failed: {str(e)}")
                image_comparison = None
        
        # Extract and compare fonts
        font_comparison = None
        if PYMUPDF_AVAILABLE:
            print("\nExtracting fonts from PDFs...")
            try:
                fonts1 = self.extract_fonts_from_pdf(pdf1_path)
                fonts2 = self.extract_fonts_from_pdf(pdf2_path)
                print(f"✓ Extracted {fonts1['unique_count']} unique fonts from {Path(pdf1_path).name}")
                print(f"✓ Extracted {fonts2['unique_count']} unique fonts from {Path(pdf2_path).name}")
                
                print("Comparing fonts...")
                font_comparison = self.compare_fonts(fonts1, fonts2)
                print(f"✓ Font comparison complete: {font_comparison['common_count']} common fonts found")
            except Exception as e:
                print(f"⚠ Warning: Font comparison failed: {str(e)}")
                font_comparison = None
        
        # Generate comparison report
        print("\nGenerating comparison report...")
        report = self.generate_comparison_report(
            text1, 
            text2, 
            Path(pdf1_path).name, 
            Path(pdf2_path).name,
            semantic_sim_max,
            semantic_sim_avg,
            text_diff,
            image_comparison,
            font_comparison
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
