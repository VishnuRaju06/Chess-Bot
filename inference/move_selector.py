import chess
import numpy as np
import torch

from inference.board_encoder import board_to_tensor

def select_bot_move(
    board: chess.Board,
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 0.7,
    top_k: int | None = 5,
) -> chess.Move:
    """
    Selects a move using masked policy head + temperature sampling.
    Matches Colab inference exactly.
    """
    x = board_to_tensor(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)[0].cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_indices = [
        move.from_square * 64 + move.to_square
        for move in legal_moves
    ]

    legal_logits = logits[legal_indices]

    # 🔒 NUMERICAL STABILITY (IMPORTANT)
    legal_logits -= legal_logits.max()

    # Optional top-k filter to avoid very low-probability blunders.
    if top_k is not None and top_k < len(legal_moves):
        top_idxs = np.argpartition(-legal_logits, top_k)[:top_k]
        legal_moves = [legal_moves[i] for i in top_idxs]
        legal_logits = legal_logits[top_idxs]

    if temperature > 0:
        probs = np.exp(legal_logits / temperature)
        probs /= probs.sum()
        idx = np.random.choice(len(legal_moves), p=probs)
    else:
        idx = np.argmax(legal_logits)

    return legal_moves[idx]
