import torch
from huggingface_hub import hf_hub_download
from inference.model import ChessPolicyNet


class ChessInference:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = ChessPolicyNet().to(self.device)

        model_path = hf_hub_download(
            repo_id="VishnuRaju06/chess-bot-model",
            filename="chess_model.pth"
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.eval()