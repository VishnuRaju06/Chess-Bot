import chess
import torch

from inference.predict import ChessInference
from inference.move_selector import select_bot_move


def main():
    board = chess.Board()

    # Load model
    engine = ChessInference("model/model.pth")
    model = engine.model
    device = engine.device

    # Select bot move
    move = select_bot_move(
        board,
        model,
        device,
        temperature=0.0   # 0 = strongest (argmax)
    )

    print("AI move:", move.uci())


if __name__ == "__main__":
    main()
