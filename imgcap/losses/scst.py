import torch

from imgcap.metrics.eval_metrics import eval_CIDEr

_SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<EN>"}


def _ids_to_words(ids_tensor, vocab_itos):
    result = []
    for seq in ids_tensor.tolist():
        words = []
        for idx in seq:
            w = vocab_itos.get(idx, "<unk>")
            if w == "<eos>":
                break
            if w not in _SPECIAL_TOKENS:
                words.append(w)
        result.append(words)
    return result


def scst_loss(log_probs, sampled_ids, greedy_ids, references, vocab_itos):
    """Self-Critical Sequence Training loss (Rennie et al., 2017).

    The reward is CIDEr(sampled caption) - CIDEr(greedy caption): the
    greedy decode acts as the REINFORCE baseline, so the policy only gets
    pushed toward samples that beat its own greedy behaviour.

    Args:
        log_probs   : (B, T)  -- from Decoder.sample_with_log_probs
        sampled_ids : (B, T)  -- sampled token indices
        greedy_ids  : (B, T)  -- greedy token indices (from Decoder.greedy_rollout)
        references  : List[List[List[str]]]  -- same format as eval_CIDEr
        vocab_itos  : dict
    """
    sampled_captions = _ids_to_words(sampled_ids, vocab_itos)
    greedy_captions = _ids_to_words(greedy_ids, vocab_itos)

    rewards = []
    for s, g, r in zip(sampled_captions, greedy_captions, references):
        rs, _ = eval_CIDEr([s], [r])
        rg, _ = eval_CIDEr([g], [r])
        rewards.append(rs - rg)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=log_probs.device)  # (B,)

    # Padding positions already have log_prob == 0 (frozen after <eos>), so
    # summing over time and weighting by the per-image reward gives the
    # standard REINFORCE policy-gradient loss.
    seq_log_prob = log_probs.sum(dim=1)  # (B,)
    loss = -(rewards * seq_log_prob).mean()
    return loss
