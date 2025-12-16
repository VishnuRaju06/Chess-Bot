import chess

from inference.move_selector import select_bot_move


class ChessGame:
    """
    Game wrapper that keeps the board, move history and knows which side
    the AI is playing. Exposes small helpers used by both the CLI tests
    and the HTTP API layer.
    """

    def __init__(self, engine, ai_color=chess.BLACK):
        self.engine = engine
        self.ai_color = ai_color
        self.human_color = chess.WHITE if ai_color == chess.BLACK else chess.BLACK
        self.board = chess.Board()
        self.move_history: list[str] = []

    def reset(self):
        self.board.reset()
        self.move_history.clear()

    def get_fen(self) -> str:
        return self.board.fen()

    def apply_human_move(self, move_uci: str):
        """Apply a human move (assumed to be the opposite color of the AI)."""
        move = chess.Move.from_uci(move_uci)

        if self.board.turn != self.human_color:
            raise ValueError("Not human turn")

        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")

        self.board.push(move)
        self.move_history.append(move_uci)

    def apply_ai_move(self, temperature: float = 0.3, top_k: int | None = 5) -> str:
        """Select and apply the AI move for the configured AI color."""
        if self.board.is_game_over():
            raise ValueError("Game is already over")

        if self.board.turn != self.ai_color:
            # If called out of turn, simply no-op to avoid corrupting state.
            return ""

        move = select_bot_move(
            self.board,
            self.engine.model,
            self.engine.device,
            temperature=temperature,
            top_k=top_k,
        )
        self.board.push(move)
        uci = move.uci()
        self.move_history.append(uci)
        return uci

    def state(self):
        legal_moves = {}
        for move in self.board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            legal_moves.setdefault(from_sq, []).append(to_sq)

        return {
            "fen": self.board.fen(),
            "move_history": self.move_history,
            "game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None,
            "legal_moves": legal_moves,
        }
