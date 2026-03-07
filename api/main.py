import os
from pathlib import Path
from typing import Final

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from game.game_manager import ChessGame
from inference.predict import ChessInference

app = FastAPI()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
FRONTEND_DIR: Final[Path] = BASE_DIR / "frontend"
DEFAULT_MODEL_PATH: Final[Path] = BASE_DIR / "model" / "model.pth"

# Allow overriding model path / download URL via environment variables for deployment
MODEL_PATH_ENV: Final[str | None] = os.getenv("MODEL_PATH")
MODEL_URL_ENV: Final[str | None] = os.getenv("MODEL_URL")

MODEL_PATH: Final[Path] = (
    Path(MODEL_PATH_ENV) if MODEL_PATH_ENV else DEFAULT_MODEL_PATH
)

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static",
)
# Serve piece sprites (kept outside /static)
app.mount(
    "/pieces",
    StaticFiles(directory=str(FRONTEND_DIR / "pieces")),
    name="pieces",
)

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


def _ensure_model_file() -> Path:
    """
    Ensure the model file exists at MODEL_PATH.

    If it does not exist but MODEL_URL is provided, download it there.
    """
    model_path = MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        return model_path

    if MODEL_URL_ENV:
        # Lazy import so local development without MODEL_URL_ENV does not require requests
        import requests

        try:
            response = requests.get(MODEL_URL_ENV, stream=True, timeout=60)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download model from {MODEL_URL_ENV!r}: {exc}"
            ) from exc

        with model_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return model_path

    raise FileNotFoundError(
        f"Model not found at {model_path}. "
        "Either place your trained weights there, "
        "set MODEL_PATH to a valid local path, "
        "or set MODEL_URL to a downloadable model URL."
    )


# ------------------ Game wiring ------------------ #
resolved_model_path = _ensure_model_file()

engine = ChessInference(str(resolved_model_path))
print(f"[startup] Loaded model from {resolved_model_path} on device {engine.device}")
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
