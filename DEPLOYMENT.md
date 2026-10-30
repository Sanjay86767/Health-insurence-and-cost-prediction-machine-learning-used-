# 🚀 Production Cloud Deployment Guide
## Medical Health Insurance Cost Prediction & Counterfactual Recourse

This guide provides step-by-step instructions to deploy this project publicly to the cloud so anyone with a web link can use it.

---

## Option 1: Streamlit Community Cloud (100% Free & Fastest — Recommended for College Submission)

Streamlit Community Cloud is free for public GitHub repositories and requires zero server configuration.

### Steps:
1. **Initialize Git & Push to GitHub:**
   ```powershell
   cd "c:\Users\HP\OneDrive\Desktop\health insurence"
   git init
   git add .
   git commit -m "Initial commit of Explainable Health Insurance ML"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/health-insurance-ml.git
   git push -u origin main
   ```
2. **Deploy on Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"Create app"** / **"New app"**.
   - Select your repository: `<YOUR_GITHUB_USERNAME>/health-insurance-ml`.
   - Set **Branch:** `main`.
   - Set **Main file path:** `app.py`.
   - Click **"Deploy!"**.
3. **Result:** Your app will be live within 2 minutes at a public URL:
   `https://health-insurance-ml-<random>.streamlit.app`

---

## Option 2: Render (Free Web Service)

1. Sign up for free at [render.com](https://render.com/).
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Connect your GitHub repository.
4. Render will auto-detect `render.yaml` or use these settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## Option 3: Hugging Face Spaces (Free CPU Tier)

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **Spaces** $\rightarrow$ **"Create new Space"**.
3. Select **Space SDK:** `Streamlit`.
4. Upload project files or link your GitHub repo.
5. Hugging Face will automatically build dependencies from `requirements.txt` and launch your app.

---

## Option 4: Local / Cloud Docker Container

To run in a containerized environment (AWS ECS, GCP Cloud Run, or local Docker Desktop):

```bash
# 1. Build the Docker image
docker build -t health-insurance-ml:latest .

# 2. Run the container exposing ports 8501 (UI) and 8000 (FastAPI)
docker run -p 8501:8501 -p 8000:8000 health-insurance-ml:latest

# 3. Access in browser
# Streamlit UI: http://localhost:8501
# FastAPI Docs: http://localhost:8000/docs
```
