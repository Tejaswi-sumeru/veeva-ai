# Streamlit Cloud Deployment Guide

## Quick Deploy Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: PDF Comparison Tool"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

## Deployment Checklist

### ✅ Pre-Deployment

- [x] `requirements.txt` includes all dependencies
- [x] `.streamlit/config.toml` is configured
- [x] `app.py` is in the root directory
- [x] No hardcoded file paths
- [x] No API keys or secrets required
- [x] All imports are available in requirements.txt

### ✅ Files Required for Deployment

- `app.py` - Main Streamlit application
- `compare_pdfs.py` - Core comparison logic
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration (optional but recommended)

### ✅ Dependencies Verified

All required packages are in `requirements.txt`:
- `streamlit>=1.28.0` - Web framework
- `sentence-transformers>=2.2.0` - Hugging Face models
- `scikit-learn>=1.0.0` - ML utilities
- `PyPDF2>=3.0.0` - PDF text extraction
- `numpy>=1.21.0` - Numerical computing
- `PyMuPDF>=1.23.0` - PDF highlighting

### ✅ Configuration

The app is configured with:
- Wide layout for better viewing
- Custom theme colors
- Session state management
- Error handling for missing dependencies
- Temporary file management

## Post-Deployment

### First Run

- The app will download the Hugging Face model on first use
- This may take 1-2 minutes
- Subsequent runs will be faster (model is cached)

### Monitoring

- Check Streamlit Cloud logs for any errors
- Monitor resource usage (memory/CPU)
- Test with sample PDFs to verify functionality

## Troubleshooting

### App Won't Deploy

1. Check that `app.py` is in the root directory
2. Verify `requirements.txt` has no syntax errors
3. Ensure all dependencies are available on PyPI

### Model Download Fails

1. Check internet connectivity in Streamlit Cloud
2. Verify Hugging Face model name is correct
3. Check Streamlit Cloud logs for specific errors

### Memory Issues

- Large PDFs may consume significant memory
- Consider adding file size limits in the UI
- Monitor Streamlit Cloud resource usage

## Local Testing Before Deployment

Test locally first:
```bash
streamlit run app.py
```

Verify:
- [ ] PDF upload works
- [ ] Comparison completes successfully
- [ ] Report generation works
- [ ] PDF highlighting works
- [ ] Download buttons work

## Notes

- No environment variables needed (uses free Hugging Face models)
- No API keys required
- Models are downloaded automatically on first use
- All processing happens server-side

