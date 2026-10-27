import os
from threading import RLock
from uuid import uuid4
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

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Vercel will have dynamic URLs)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    repo_id="vishnuraju06/Chess-Bot",
    filename="model.pth"
)
engine = ChessInference(MODEL_PATH)
print(f"[startup] Loaded model from Hugging Face on device {engine.device}")

# Each browser gets an independent game and lock. The shared model is read-only
# in eval mode, while board mutations stay serialized per game.
games: dict[str, ChessGame] = {}
game_locks: dict[str, RLock] = {}
games_lock = RLock()


# ------------------ API Models ------------------ #

class MoveRequest(BaseModel):
    game_id: str
    move: str


class StartRequest(BaseModel):
    color: str = "white"


# ------------------ API Endpoints ------------------ #

@app.post("/start")
def start_game(req: StartRequest):
    if req.color not in ("white", "black"):
        raise HTTPException(status_code=400, detail="color must be white or black")

    game_id = uuid4().hex
    new_game = ChessGame(engine)
    new_game.reset(human_color=req.color)
    game_lock = RLock()
    with game_lock:
        if new_game.board.turn == new_game.ai_color:
            new_game.apply_ai_move()
        state = new_game.state()
    with games_lock:
        games[game_id] = new_game
        game_locks[game_id] = game_lock
    return {"game_id": game_id, **state}


def get_game(game_id: str) -> tuple[ChessGame, RLock]:
    with games_lock:
        game = games.get(game_id)
        game_lock = game_locks.get(game_id)
    if game is None or game_lock is None:
        raise HTTPException(status_code=404, detail="Game not found. Start a new game.")
    return game, game_lock


@app.get("/state")
def get_state(game_id: str):
    game, game_lock = get_game(game_id)
    with game_lock:
        return {"game_id": game_id, **game.state()}


@app.post("/move")
def make_move(req: MoveRequest):
    game, game_lock = get_game(req.game_id)
    try:
        with game_lock:
            game.apply_human_move(req.move)
            if not game.board.is_game_over():
                game.apply_ai_move()
            state = game.state()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"game_id": req.game_id, **state}


@app.post("/ai-move")
def ai_move(game_id: str):
    game, game_lock = get_game(game_id)
    try:
        with game_lock:
            if not game.board.is_game_over() and game.board.turn == game.ai_color:
                game.apply_ai_move()
            state = game.state()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"game_id": game_id, **state}
