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
        self.move_history: list[dict] = []

    def set_human_color(self, color: str):
        if color == "black":
            self.human_color = chess.BLACK
            self.ai_color = chess.WHITE
        else:
            self.human_color = chess.WHITE
            self.ai_color = chess.BLACK

    def reset(self, human_color: str | None = None):
        if human_color is not None:
            self.set_human_color(human_color)
        self.board.reset()
        self.move_history.clear()

    def get_fen(self) -> str:
        return self.board.fen()

    def _push(self, move: chess.Move) -> str:
        record = {"uci": move.uci(), "san": self.board.san(move)}
        self.board.push(move)
        self.move_history.append(record)
        return record["uci"]

    def apply_human_move(self, move_uci: str):
        """Apply a human move (assumed to be the opposite color of the AI)."""
        move = chess.Move.from_uci(move_uci)

        if self.board.turn != self.human_color:
            raise ValueError("Not human turn")

        if move not in self.board.legal_moves:
            raise ValueError("Illegal move")

        self._push(move)

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
        return self._push(move)

    def state(self):
        legal_moves = {}
        for move in self.board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            promo = None
            if move.promotion:
                promo = chess.piece_symbol(move.promotion).lower()
            legal_moves.setdefault(from_sq, []).append(
                {
                    "to": chess.square_name(move.to_square),
                    "promotion": promo,
                    "uci": move.uci(),
                }
            )

        outcome = self.board.outcome()
        return {
            "fen": self.board.fen(),
            "move_history": self.move_history,
            "game_over": self.board.is_game_over(),
            "result": outcome.result() if outcome else None,
            "termination": outcome.termination.name if outcome else None,
            "in_check": self.board.is_check(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "ai_color": "white" if self.ai_color == chess.WHITE else "black",
            "legal_moves": legal_moves,
        }
