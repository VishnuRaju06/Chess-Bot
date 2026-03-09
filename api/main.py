import os
from pathlib import Path
from typing import Final

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from game.game_manager import ChessGame
from inference.predict import ChessInference

app = FastAPI()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
FRONTEND_DIR: Final[Path] = BASE_DIR / "frontend"

# ------------------ Serve frontend ------------------ #

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static",
)

app.mount(
    "/pieces",
    StaticFiles(directory=str(FRONTEND_DIR / "pieces")),
    name="pieces",
)

@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# ------------------ Load model from Hugging Face ------------------ #

MODEL_PATH = hf_hub_download(
    repo_id="VishnuRaju06/chess-bot-model",
    filename="chess_model.pth"
)

engine = ChessInference(MODEL_PATH)
print(f"[startup] Loaded model from Hugging Face on device {engine.device}")

game = ChessGame(engine, ai_color=chess.BLACK)


# ------------------ API Models ------------------ #

class MoveRequest(BaseModel):
    move: str


# ------------------ API Endpoints ------------------ #

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