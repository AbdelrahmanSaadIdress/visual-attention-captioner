import pandas as pd
import spacy

_SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<EN>"}


def collect_references_per_image(split_csv_path: str, split: str, vocab=None,
                                   spacy_model_name: str = "en_core_web_sm") -> dict:
    """Build {image_id: [[tok, tok, ...], ...]} for every image in `split`.

    Tokenized the same way `Flickr30._tokenize` does, with special tokens
    stripped. `vocab` is accepted (and unused) for call-site compatibility
    with the original notebooks.
    """
    df = pd.read_csv(split_csv_path)
    df = df[df["split"] == split]

    spacy_model = spacy.load(spacy_model_name)

    refs_dict: dict = {}
    for _, row in df.iterrows():
        image_id = row["image_name"]
        caption = row["comment"] if isinstance(row["comment"], str) else ""
        words = [
            f" {tok.text.lower()}" for tok in spacy_model(caption)
            if tok.text.lower() not in _SPECIAL_TOKENS
        ]
        refs_dict.setdefault(image_id, []).append(words)

    return refs_dict
