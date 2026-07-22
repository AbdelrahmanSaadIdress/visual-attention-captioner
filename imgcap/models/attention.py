import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over CNN spatial features.

    Projects features and hidden state independently into a shared
    attention_dim space before combining -- fewer parameters and more
    stable than concatenating and projecting the full feature+hidden vector.
    """

    def __init__(self, hidden_size: int = 512, feature_size: int = 2048,
                 attention_dim: int = 512):
        super().__init__()
        self.feat_proj = nn.Linear(feature_size, attention_dim)
        self.hidden_proj = nn.Linear(hidden_size, attention_dim)
        self.energy_proj = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        """
        Args:
            features     : (B, num_pixels, feature_size)   e.g. (B, 49, 2048)
            hidden_state : (B, hidden_size)                last layer of LSTM

        Returns:
            context      : (B, feature_size)   weighted sum of features
            attn_weights : (B, num_pixels)     attention distribution
        """
        feat_energy = self.feat_proj(features)                       # (B, 49, attn_dim)
        hidden_energy = self.hidden_proj(hidden_state).unsqueeze(1)  # (B,  1, attn_dim)

        energy = torch.tanh(feat_energy + hidden_energy)             # (B, 49, attn_dim)
        scores = self.energy_proj(energy).squeeze(2)                 # (B, 49)
        attn = F.softmax(scores, dim=1)                               # (B, 49)
        context = torch.sum(features * attn.unsqueeze(2), dim=1)      # (B, feature_size)
        return context, attn
