from inference.predict import ChessInference
from game.game_manager import ChessGame
import chess


def main():
    engine = ChessInference("model/model.pth")

    # Human = White, AI = Black
    game = ChessGame(engine, ai_color=chess.BLACK)

    print("Initial FEN:", game.get_fen())

    # Human move
    game.apply_human_move("e2e4")
    print("After human move:", game.get_fen())

    # AI move
    ai_move = game.apply_ai_move()
    print("AI move:", ai_move)
    print("After AI move:", game.get_fen())

    print("Move history:", game.move_history)


if __name__ == "__main__":
    main()
