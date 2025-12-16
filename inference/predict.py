import torch
from inference.model import ChessPolicyNet

class ChessInference:
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = ChessPolicyNet().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
