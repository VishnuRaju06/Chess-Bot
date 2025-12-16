import os

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from inference.predict import ChessInference
from game.game_manager import ChessGame
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")),
    name="static"
)
# Serve piece sprites (kept outside /static)
app.mount(
    "/pieces",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "pieces")),
    name="pieces"
)

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ------------------ Game wiring ------------------ #
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please place your trained weights there.")

engine = ChessInference(MODEL_PATH)
print(f"[startup] Loaded model from {MODEL_PATH} on device {engine.device}")
game = ChessGame(engine, ai_color=chess.BLACK)


class MoveRequest(BaseModel):
    move: str


@app.post("/start")
def start_game():
    game.reset()
    return game.state()


@app.get("/state")
def get_state():
    return game.state()


@app.post("/move")
def make_move(req: MoveRequest):
    try:
        game.apply_human_move(req.move)
        if not game.board.is_game_over():
            game.apply_ai_move()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return game.state()
