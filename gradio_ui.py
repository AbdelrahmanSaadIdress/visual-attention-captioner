"""
Gradio demo for the ImgCap (ResNet50 encoder + attention + LSTM decoder) model.

- Feeds the decoder with <sos> followed by <EN> before generation starts
  (matching how the model was trained: captions = [<sos>, <EN>, ...tokens..., <eos>]).
- Uses beam search decoding instead of greedy argmax.
- Loads weights/vocab either from local files or straight from the
  Hugging Face Hub repo (AbdoSaad24/image-captioning_production_ready).

Run:
    pip install gradio torch torchvision huggingface_hub pillow
    python app.py
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Optional

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision import transforms as T

# ─────────────────────────────────────────────────────────────────────────
# Config -- edit these if your repo id / filenames differ
# ─────────────────────────────────────────────────────────────────────────
HF_REPO_ID = "AbdoSaad24/image-captioning_production_ready"
CHECKPOINT_FILENAME = "best_model.pt"
VOCAB_STOI_FILENAME = "vocab_stoi.pkl"
VOCAB_ITOS_FILENAME = "vocab_itos.pkl"

# Local paths -- since you already downloaded these, point straight at them.
# Change these if your local files live somewhere else (e.g. just
# "best_model.pt" if they're next to app.py).
LOCAL_CHECKPOINT_PATH = "production_needs/best_model.pt"
LOCAL_VOCAB_STOI_PATH = "production_needs/vocab_stoi.pkl"
LOCAL_VOCAB_ITOS_PATH = "production_needs/vocab_itos.pkl"

# Model hyper-params -- must match training (configs/default.yaml / scst.yaml)
FEATURE_SIZE = 2048
LSTM_HIDDEN_SIZE = 1024
NUM_LAYERS = 2
EMBEDDING_DIM = 1024
ATTENTION_DIM = 512
DROPOUT = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<EN>", "<unk>"}


# ─────────────────────────────────────────────────────────────────────────
# Model definition (same architecture as imgcap/models/*)
# ─────────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int, attention_dim: int):
        super().__init__()
        self.feat_proj = nn.Linear(feature_size, attention_dim)
        self.hidden_proj = nn.Linear(hidden_size, attention_dim)
        self.energy_proj = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        feat_energy = self.feat_proj(features)
        hidden_energy = self.hidden_proj(hidden_state).unsqueeze(1)
        energy = torch.tanh(feat_energy + hidden_energy)
        scores = self.energy_proj(energy).squeeze(2)
        attn = F.softmax(scores, dim=1)
        context = torch.sum(features * attn.unsqueeze(2), dim=1)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, feature_size, lstm_hidden_size, num_layers, vocab_size,
                 embedding_dim, attention_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(lstm_hidden_size, feature_size, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=feature_size + embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(lstm_hidden_size, vocab_size)
        self.init_h = nn.Linear(feature_size, lstm_hidden_size)
        self.init_c = nn.Linear(feature_size, lstm_hidden_size)

    def init_hidden_state(self, features):
        mean_features = features.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean_features))
        c0 = torch.tanh(self.init_c(mean_features))
        num_layers = self.lstm.num_layers
        batch_size = features.size(0)
        device = features.device
        h = torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device)
        h[0] = h0
        c[0] = c0
        return h, c

    def step(self, input_word, h, c, features):
        """Single decoding step. input_word: (B,) token ids."""
        context, _ = self.attention(features, h[-1])
        embeddings = self.embedding(input_word)
        lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
        out, (h, c) = self.lstm(lstm_input, (h, c))
        logits = self.fc(out.squeeze(1))
        return logits, h, c


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.ResNet50 = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(self.ResNet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)  # (B, 49, 2048)
        return x


class ImgCap(nn.Module):
    def __init__(self, feature_size, lstm_hidden_size, num_layers, vocab_size,
                 embedding_dim, attention_dim, dropout):
        super().__init__()
        self.encoder = ResNet50()
        self.decoder = Decoder(feature_size, lstm_hidden_size, num_layers, vocab_size,
                                embedding_dim, attention_dim, dropout)


# ─────────────────────────────────────────────────────────────────────────
# Beam search, seeded with <sos> then <EN>
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Beam:
    tokens: List[int]
    log_prob: float
    h: torch.Tensor
    c: torch.Tensor


def beam_search_caption(model, image_tensor, vocab_stoi, vocab_itos,
                         device, beam_width: int = 5, max_seq_length: int = 50):
    """Beam search decoding that primes the decoder with <sos>, <EN> before
    letting the model generate freely (matches training-time caption format:
    [<sos>, <EN>, tok1, tok2, ..., <eos>])."""
    model.eval()
    sos_idx = vocab_stoi["<sos>"]
    en_idx = vocab_stoi["<EN>"]
    end_idx = vocab_stoi["<eos>"]

    with torch.no_grad():
        features = model.encoder(image_tensor.to(device))  # (1, 49, 2048)
        h0, c0 = model.decoder.init_hidden_state(features)

        # ── Prime step: feed <sos>, then feed <EN> (not scored/branched) ──
        sos_tok = torch.tensor([sos_idx], dtype=torch.long, device=device)
        _, h1, c1 = model.decoder.step(sos_tok, h0, c0, features)

        en_tok = torch.tensor([en_idx], dtype=torch.long, device=device)
        _, h2, c2 = model.decoder.step(en_tok, h1, c1, features)

        beams: List[Beam] = [Beam(tokens=[sos_idx, en_idx], log_prob=0.0, h=h2, c=c2)]
        completed: List[Beam] = []

        for _ in range(max_seq_length):
            if not beams:
                break
            candidates: List[Beam] = []
            for beam in beams:
                last_token = torch.tensor([beam.tokens[-1]], dtype=torch.long, device=device)
                logits, h_n, c_n = model.decoder.step(last_token, beam.h, beam.c, features)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_tokens = log_probs.topk(beam_width)

                for log_p, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                    new_beam = Beam(
                        tokens=beam.tokens + [token],
                        log_prob=beam.log_prob + log_p,
                        h=h_n, c=c_n,
                    )
                    if token == end_idx:
                        completed.append(new_beam)
                    else:
                        candidates.append(new_beam)

            candidates.sort(key=lambda bm: bm.log_prob / len(bm.tokens), reverse=True)
            beams = candidates[:beam_width]

        if not completed:
            completed = beams

        best = max(completed, key=lambda bm: bm.log_prob / len(bm.tokens))
        words = [vocab_itos[t] for t in best.tokens if vocab_itos[t] not in _SPECIAL_TOKENS]

    return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────
# Loading model + vocab (local files if present, else pull from HF Hub)
# ─────────────────────────────────────────────────────────────────────────
def _resolve_path(local_name: str, hf_filename: str) -> str:
    import os
    if os.path.exists(local_name):
        print(f"  found locally -> {local_name}")
        return local_name
    print(f"  not found locally, downloading {hf_filename} from {HF_REPO_ID} ...")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename)


def load_everything():
    ckpt_path = _resolve_path(LOCAL_CHECKPOINT_PATH, CHECKPOINT_FILENAME)
    stoi_path = _resolve_path(LOCAL_VOCAB_STOI_PATH, VOCAB_STOI_FILENAME)
    itos_path = _resolve_path(LOCAL_VOCAB_ITOS_PATH, VOCAB_ITOS_FILENAME)

    with open(stoi_path, "rb") as f:
        vocab_stoi = pickle.load(f)
    with open(itos_path, "rb") as f:
        vocab_itos = pickle.load(f)

    model = ImgCap(
        feature_size=FEATURE_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        vocab_size=len(vocab_stoi),
        embedding_dim=EMBEDDING_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab_stoi, vocab_itos


print("Loading model & vocab ...")
MODEL, VOCAB_STOI, VOCAB_ITOS = load_everything()
print("Ready.")

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(image: Image.Image, beam_width: int):
    if image is None:
        return "Please upload an image."
    img = image.convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0)
    caption = beam_search_caption(
        MODEL, img_tensor, VOCAB_STOI, VOCAB_ITOS, DEVICE,
        beam_width=int(beam_width),
    )
    return caption


with gr.Blocks(title="Image Captioning") as demo:
    gr.Markdown("# 🖼️ Image Captioning (ResNet50 + Attention + LSTM, beam search)")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an image")
            beam_width_input = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                          label="Beam width")
            run_btn = gr.Button("Generate caption", variant="primary")
        with gr.Column():
            caption_output = gr.Textbox(label="Generated caption", lines=3)

    run_btn.click(fn=predict, inputs=[image_input, beam_width_input], outputs=caption_output)
    image_input.change(fn=predict, inputs=[image_input, beam_width_input], outputs=caption_output)

if __name__ == "__main__":
    demo.launch()