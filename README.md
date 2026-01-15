# PDF Comparison Tool using Hugging Face Models

A web application that compares two PDF documents using free Hugging Face models. Features a Streamlit web interface for easy PDF comparison with semantic similarity analysis and visual difference highlighting.

## Features

- ✅ **Free & Open Source**: Uses free Hugging Face models (no API keys required)
- ✅ **Web Interface**: Beautiful Streamlit UI for easy PDF comparison
- ✅ **Semantic Similarity**: Uses sentence-transformers for intelligent comparison
- ✅ **Comprehensive Analysis**:
  - Summary of both documents
  - Semantic similarity scores
  - Key differences (added, removed, changed lines)
  - Word-level analysis
  - Section-by-section comparison
  - Overall conclusion
- ✅ **Image Comparison**: 
  - Extract and compare images/logos from PDFs
  - Perceptual hashing for visual similarity detection
  - Side-by-side image preview
  - Image similarity scores
- ✅ **Font Comparison**:
  - Extract font information from PDFs
  - Compare fonts between documents
  - Identify common and unique fonts
  - Font similarity analysis
- ✅ **Visual Highlights**: Generate highlighted PDFs showing differences
- ✅ **Automatic Text Truncation**: Handles large PDFs efficiently
- ✅ **Robust Error Handling**: Validates files and handles edge cases

## Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading Hugging Face models on first use)

## Installation

### Local Development

1. **Clone or navigate to this directory:**
   ```bash
   cd veeva-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - The app will open in your browser at `http://localhost:8501`

### Command Line Usage

You can also use the comparison tool from the command line:

```bash
python compare_pdfs.py <pdf1_path> <pdf2_path>
```

**Example:**
```bash
python compare_pdfs.py document_v1.pdf document_v2.pdf
```

## Streamlit Cloud Deployment

### Deploy to Streamlit Cloud

1. **Push your code to GitHub:**
   - Create a GitHub repository
   - Push this code to the repository

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to: `app.py`
   - Click "Deploy"

3. **That's it!** Your app will be live in minutes.

### Deployment Requirements

- ✅ `requirements.txt` is already configured
- ✅ `.streamlit/config.toml` is configured for production
- ✅ No API keys or secrets needed (uses free Hugging Face models)
- ✅ All dependencies are specified in `requirements.txt`

### Streamlit Cloud Configuration

The app is pre-configured with:
- Theme settings in `.streamlit/config.toml`
- Server settings for production
- Browser settings optimized for deployment

## Usage

### Web Interface

1. **Upload PDFs:**
   - Upload Document 1 (Original)
   - Upload Document 2 (To Compare)

2. **Click "Compare PDFs":**
   - The app will extract text from both PDFs
   - Calculate semantic similarity
   - Analyze differences
   - Generate a comprehensive report

3. **View Results:**
   - **Report Tab**: Full text comparison report
   - **Line Differences Tab**: Detailed line-by-line differences with filtering
   - **Images & Fonts Tab**: Visual image comparison and font analysis
   - **Highlighted PDF Tab**: Download PDF2 with visual highlights showing differences

### Output

The comparison provides:
- Semantic similarity percentage
- Number of added, removed, and changed lines
- Word overlap statistics
- Detailed line-by-line differences
- Section-by-section comparison
- Downloadable report and highlighted PDF

## How It Works

1. **File Validation**: Checks that both PDF files exist and are valid
2. **Text Extraction**: Extracts text from both PDFs using PyPDF2 (local processing)
3. **Text Chunking**: Splits text into overlapping chunks for better comparison
4. **Semantic Analysis**: Uses Hugging Face sentence-transformers to:
   - Generate embeddings for text chunks
   - Calculate cosine similarity between documents
   - Identify semantic similarities
5. **Difference Detection**: Uses Python's difflib to find:
   - Added lines
   - Removed lines
   - Changed lines
   - Word-level differences
6. **Report Generation**: Creates comprehensive comparison report
7. **PDF Highlighting**: Uses PyMuPDF to highlight differences in PDF2

## File Size Limits

- **No strict file size limit** for PDFs (text is extracted locally)
- Text is automatically truncated to ~500,000 characters per PDF for processing efficiency
- Very large PDFs will have their content truncated, but comparison will still work
- For best results with large documents, consider splitting them into sections

## Error Handling

The application handles various error cases:
- Missing or invalid PDF files
- Corrupted PDFs
- Text extraction failures
- Network errors during model download
- Large file handling (automatic truncation)
- Empty or password-protected PDFs

## Cost Considerations

- ✅ **100% Free**: Uses free Hugging Face models
- ✅ **No API Keys Required**: Everything runs locally after initial model download
- ✅ **No Usage Limits**: Process as many PDFs as you need
- ✅ **Offline Capable**: After first use, models are cached locally

## Troubleshooting

### "PyPDF2 not installed"
- Install PyPDF2: `pip install PyPDF2`
- Or install all requirements: `pip install -r requirements.txt`

### "sentence-transformers not installed"
- Install: `pip install sentence-transformers scikit-learn`
- Or install all requirements: `pip install -r requirements.txt`

### "Text extraction failed"
- Ensure the PDF files are not corrupted
- Check that the PDFs are not password-protected
- Try with different PDF files to isolate the issue

### Model Download Issues
- Ensure you have an internet connection for the first run
- Models are cached locally after first download
- Check your firewall settings if downloads fail

### Streamlit Cloud Deployment Issues
- Ensure `requirements.txt` includes all dependencies
- Check that `app.py` is in the root directory
- Verify Python version compatibility (3.8+)

## Requirements

- `sentence-transformers>=2.2.0` - Hugging Face sentence transformers
- `scikit-learn>=1.0.0` - Machine learning utilities
- `PyPDF2>=3.0.0` - PDF text extraction
- `numpy>=1.21.0` - Numerical computing
- `streamlit>=1.28.0` - Web framework
- `PyMuPDF>=1.23.0` - PDF manipulation, image extraction, and font extraction
- `Pillow>=9.0.0` - Image processing
- `imagehash>=4.3.1` - Perceptual image hashing for image comparison

## Project Structure

```
veeva-ai/
├── app.py                 # Streamlit web application
├── compare_pdfs.py        # Core comparison logic
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## License

This project is provided as-is for educational and development purposes.

