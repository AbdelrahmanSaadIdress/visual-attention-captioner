from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


def greedy_search(model, images, vocab, device="cpu",
                   start_token="<sos>", end_token="<eos>", max_seq_length=50):
    model.eval()
    images = images.to(device)
    start_idx = vocab.vocab_stoi[start_token]
    end_idx = vocab.vocab_stoi[end_token]

    with torch.no_grad():
        features = model.encoder(images)
        batch_size = features.size(0)
        input_word = torch.full((batch_size,), start_idx, dtype=torch.long, device=device)
        h, c = model.decoder.init_hidden_state(features)
        captions = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_seq_length):
            context, _ = model.decoder.attention(features, h[-1])
            embeddings = model.decoder.embedding(input_word)
            lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
            out, (h, c) = model.decoder.lstm(lstm_input, (h, c))
            output = model.decoder.fc(out.squeeze(1))
            input_word = output.argmax(1)

            for i in range(batch_size):
                if not finished[i]:
                    token = input_word[i].item()
                    word = vocab.vocab_itos[token]
                    captions[i].append(word)
                    if token == end_idx:
                        finished[i] = True
            if all(finished):
                break

    return captions


@dataclass
class Beam:
    tokens: List[int]
    log_prob: float
    h: torch.Tensor
    c: torch.Tensor


def beam_search(model, images, vocab, device="cpu",
                 start_token="<sos>", end_token="<eos>",
                 max_seq_length=50, beam_width=5):
    model.eval()
    images = images.to(device)
    start_idx = vocab.vocab_stoi[start_token]
    end_idx = vocab.vocab_stoi[end_token]

    with torch.no_grad():
        all_features = model.encoder(images)
        batch_size = all_features.size(0)
        results = []

        for b in range(batch_size):
            features = all_features[b].unsqueeze(0)
            h0, c0 = model.decoder.init_hidden_state(features)
            beams: List[Beam] = [Beam(tokens=[start_idx], log_prob=0.0, h=h0, c=c0)]
            completed: List[Beam] = []

            for _ in range(max_seq_length):
                if not beams:
                    break
                candidates: List[Beam] = []
                for beam in beams:
                    last_token = torch.tensor([beam.tokens[-1]], dtype=torch.long, device=device)
                    context, _ = model.decoder.attention(features, beam.h[-1])
                    embeddings = model.decoder.embedding(last_token)
                    lstm_input = torch.cat([context, embeddings], dim=1).unsqueeze(1)
                    out, (h_n, c_n) = model.decoder.lstm(lstm_input, (beam.h, beam.c))
                    logits = model.decoder.fc(out.squeeze(1))
                    log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                    topk_log_probs, topk_tokens = log_probs.topk(beam_width)

                    for log_p, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                        new_beam = Beam(
                            tokens=beam.tokens + [token],
                            log_prob=beam.log_prob + log_p,
                            h=h_n,
                            c=c_n,
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
            words = [vocab.vocab_itos[t] for t in best.tokens if t not in (start_idx, end_idx)]
            results.append(words)

    return results
