from typing import List

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider


def eval_BLEU(candidates: List[List[str]], references: List[List[List[str]]]) -> dict:
    """Corpus-level BLEU 1-4 with multi-reference support.

    Args:
        candidates : one hypothesis token list per image,
                     e.g. candidates[i] = ['a', 'dog', 'runs']
        references : one set of reference token lists per image (up to 5),
                     e.g. references[i] = [['a','dog','runs'], ['the','dog','is','running'], ...]

    Returns:
        dict with keys bleu_1, bleu_2, bleu_3, bleu_4
    """
    smoother = SmoothingFunction().method4

    bleu_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu_2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu_3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoother)
    bleu_4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}


def eval_CIDEr(candidates: List[List[str]], references: List[List[List[str]]]):
    """CIDEr with multi-reference support.

    Args:
        candidates : same shape as eval_BLEU
        references : same shape as eval_BLEU

    Returns:
        (avg_score, per_image_scores)
    """
    gts = {i: [" ".join(ref) for ref in references[i]] for i in range(len(references))}
    res = {i: [" ".join(candidates[i])] for i in range(len(candidates))}

    cider_scorer = Cider()
    avg_score, scores = cider_scorer.compute_score(gts=gts, res=res)
    return avg_score, scores


def token_accuracy(logits, targets, pad_idx: int = 1) -> float:
    """
    logits  : (B, T, V)
    targets : (B, T)
    Returns scalar accuracy ignoring pad positions.
    """
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()
