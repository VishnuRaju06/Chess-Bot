import chess
import numpy as np

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board into a (18, 8, 8) tensor.
    Matches the Colab preprocessing: no rank flip.
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    for piece_type, plane in PIECE_TO_PLANE.items():
        # White pieces
        for square in board.pieces(piece_type, chess.WHITE):
            r, c = divmod(square, 8)
            tensor[plane, r, c] = 1

        # Black pieces
        for square in board.pieces(piece_type, chess.BLACK):
            r, c = divmod(square, 8)
            tensor[plane + 6, r, c] = 1

    # Side to move
    tensor[12, :, :] = int(board.turn)

    # Castling rights
    tensor[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # Fullmove number (normalized)
    tensor[17, :, :] = board.fullmove_number / 100.0

    return tensor
