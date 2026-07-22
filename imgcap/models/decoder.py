import torch
import torch.nn as nn
import torch.nn.functional as F

from imgcap.models.attention import Attention


class Decoder(nn.Module):
    def __init__(self, feature_size: int = 2048, lstm_hidden_size: int = 512,
                 num_layers: int = 2, vocab_size: int = 50,
                 embedding_dim: int = 512, attention_dim: int = 512,
                 dropout: float = 0.5):
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

        # ── Image-aware hidden state initialisation ────────────────────
        # Instead of zeros, initialise h0/c0 from the mean image features so
        # the LSTM already "knows" something about the image at step 0.
        self.init_h = nn.Linear(feature_size, lstm_hidden_size)
        self.init_c = nn.Linear(feature_size, lstm_hidden_size)

    def init_hidden_state(self, features):
        """
        Args:
            features : (B, num_pixels, feature_size)
        Returns:
            h0, c0 : each (num_layers, B, hidden_size)
        """
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

    def forward(self, features, captions, teacher_forcing_ratio: float = 0.5):
        """
        Args:
            features              : (B, num_pixels, feature_size)
            captions              : (B, seq_len)   token indices including <sos>
            teacher_forcing_ratio : probability of using ground truth at each step

        Returns:
            outputs : (B, seq_len, vocab_size)
        """
        device = features.device
        batch_size = features.size(0)
        max_seq_len = captions.size(1)

        outputs = torch.zeros(batch_size, max_seq_len, self.vocab_size, device=device)
        input_word = captions[:, 0]  # <sos> token
        h, c = self.init_hidden_state(features)

        for t in range(max_seq_len):
            context, _ = self.attention(features, h[-1])
            embeddings = self.dropout(self.embedding(input_word))
            lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(out.squeeze(1)))
            outputs[:, t, :] = output

            top1 = output.argmax(1)

            # Sample the teacher-forcing decision per item in the batch,
            # not one decision for the whole batch.
            use_teacher = torch.rand(batch_size, device=device) < teacher_forcing_ratio
            input_word = torch.where(use_teacher, captions[:, t], top1)

        return outputs

    def sample_with_log_probs(self, features, vocab_stoi, max_seq_len: int = 50):
        """Multinomial sampling used by SCST's policy-gradient rollout.

        Returns:
            sampled_ids : (B, T)   -- sampled token indices
            log_probs   : (B, T)   -- log prob of each sampled token
        """
        device = features.device
        batch_size = features.size(0)
        sos_idx = vocab_stoi["<sos>"]
        eos_idx = vocab_stoi["<eos>"]

        input_word = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        h, c = self.init_hidden_state(features)

        sampled_ids = []
        log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_seq_len):
            context, _ = self.attention(features, h[-1])
            embeddings = self.embedding(input_word)
            lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            logits = self.fc(out.squeeze(1))  # (B, V)

            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            sampled = dist.sample()          # (B,)
            lp = dist.log_prob(sampled)       # (B,)

            # Once a sequence hits <eos>, freeze it.
            sampled = torch.where(finished, torch.full_like(sampled, eos_idx), sampled)
            lp = torch.where(finished, torch.zeros_like(lp), lp)

            sampled_ids.append(sampled)
            log_probs.append(lp)
            finished = finished | (sampled == eos_idx)
            input_word = sampled

            if finished.all():
                break

        sampled_ids = torch.stack(sampled_ids, dim=1)  # (B, T)
        log_probs = torch.stack(log_probs, dim=1)      # (B, T)
        return sampled_ids, log_probs

    def greedy_rollout(self, features, sos_idx: int, max_seq_len: int = 50):
        """Deterministic (argmax) rollout used as the SCST baseline.

        Returns:
            greedy_ids : (B, T)   -- argmax token indices at each step
        """
        device = features.device
        batch_size = features.size(0)

        input_word = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        h, c = self.init_hidden_state(features)

        greedy_ids = []
        for _ in range(max_seq_len):
            context, _ = self.attention(features, h[-1])
            embeddings = self.embedding(input_word)
            lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            logits = self.fc(out.squeeze(1))
            input_word = logits.argmax(1)
            greedy_ids.append(input_word)

        return torch.stack(greedy_ids, dim=1)
