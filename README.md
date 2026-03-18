## Chess-Bot

A full-stack project to play **human vs AI chess** in the browser using a custom PyTorch neural-network policy model.

### Architecture
- **Backend (FastAPI)**: Serves the chess engine logic and PyTorch inference. Deployed optimally on **Hugging Face Spaces (Docker)** for adequate RAM.
- **Frontend (HTML/JS/CSS)**: A lightweight browser UI to play chess against the AI. Deployed optimally on **Vercel**.

### What this repo contains
- **api** – FastAPI backend (`main.py`) exposing endpoints to start a game, get state, and submit moves.
- **frontend** – Static web assets, including `script.js` which safely connects to the backend API (`API_BASE`), and piece sprites.
- **game** – `ChessGame` wrapper around `python-chess` that tracks the board, move history, and wires in the PyTorch AI engine.
- **inference / model** – PyTorch policy network and inference helpers. The actual `model.pth` weights are synchronized via Git LFS.
- **notebooks** – Jupyter notebooks used to train and experiment with the model.

### Local Setup & Development
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Model Weights**
   - The bot will use the 135MB `model/model.pth` weights tracked by Git LFS, or fetch them remotely if `hf_hub_download` is configured.
3. **Start the FastAPI server**
   ```bash
   uvicorn api.main:app --reload
   ```
4. **Play**
   - Run a basic HTTP server inside the `frontend/` folder and open it in your browser, or just open `index.html` directly to test the UI. 

### Deployment

**Backend (Hugging Face Spaces):**
1. Create a **Docker Space** on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Upload this entire repository (which includes the provided `Dockerfile`). 
3. The Space will automatically construct the environment, install CPU-only PyTorch, and launch the FastAPI server natively on their free 16GB tier.

**Frontend (Vercel):**
1. Inside `frontend/static/script.js`, ensure the `API_BASE` variable is pointing squarely at your running Hugging Face Space URL.
2. Import this GitHub repository into [Vercel](https://vercel.com/).
3. Set the **Root Directory** to `frontend` during creation so Vercel only serves the UI subset.
4. Deploy!
