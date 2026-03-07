## Chess-Bot

Short project to play **human vs AI chess** in the browser using a small neural-network policy model.

### What this repo contains
- **api** – FastAPI backend (`main.py`) that serves the web UI and exposes endpoints to start a game, get state, and submit moves.
- **frontend** – Simple HTML/CSS/JS UI to play chess in the browser against the AI, including piece sprites in `pieces/`.
- **game** – `ChessGame` wrapper around `python-chess` that tracks the board, move history, and wires in the AI engine.
- **inference / model** – PyTorch policy network and inference helpers that load `model/model.pth` and select moves.
- **notebooks** – Jupyter notebooks (e.g. `kaggle_chess_training.ipynb`) used to train and experiment with the model.

### Setup & run (local)
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Place model weights**
   - Put your trained weights at `model/model.pth` (required by the API).
3. **Start the API server**
   ```bash
   uvicorn api.main:app --reload
   ```
4. **Play**
   - Open `http://127.0.0.1:8000` in your browser and play against the AI.

### Deploy to Render (single app)
1. **Ensure model is committed or available**
   - Either commit `model/model.pth` using Git LFS, or adjust the code to download it at startup.
2. **Push this repo to GitHub**
3. **Create a Render Web Service**
   - Go to Render, choose “New → Web Service”, and pick this GitHub repo.
   - Render will automatically read `render.yaml` and configure:
     - Build command: `pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt`
     - Start command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
4. **Open the app URL**
   - Once deployed, open the Render URL (e.g. `https://your-service.onrender.com`) and play in the browser.

