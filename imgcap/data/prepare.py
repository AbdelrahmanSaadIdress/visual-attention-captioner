import math
import os
import pickle
import string
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import spacy


def prepare_data(
    captions_csv: str,
    train_split: float,
    val_split: float,
    seed: int,
    split_csv_path: str,
    vocab_stoi_path: str,
    vocab_itos_path: str,
    cap: Optional[int] = None,
    on_artifact_saved=None,
) -> None:
    """
    Call this once before training. Produces:
        split_csv_path  -- original CSV + a 'split' column (train/val/test)
        vocab_stoi_path -- word -> index dict
        vocab_itos_path -- index -> word dict

    If all three files already exist this function is a no-op, so it's
    safe to call at the top of every run.

    `on_artifact_saved(local_path, artifact_name)` is an optional callback
    invoked after each file is written locally -- e.g. to also upload it
    to the HF Hub. See `imgcap.integrations.hf_hub`.
    """
    all_exist = (
        os.path.exists(split_csv_path)
        and os.path.exists(vocab_stoi_path)
        and os.path.exists(vocab_itos_path)
    )
    if all_exist:
        print("[prepare_data] All files already exist -- skipping.")
        print(f"  split csv  : {split_csv_path}")
        print(f"  vocab stoi : {vocab_stoi_path}")
        print(f"  vocab itos : {vocab_itos_path}")
        return

    print("[prepare_data] Running for the first time ...")
    os.makedirs(os.path.dirname(split_csv_path) or ".", exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(captions_csv, delimiter="|")
    df.columns = df.columns.str.strip()

    # Known bad row in the raw flickr30k results.csv (comment_number is
    # missing/malformed for this index).
    if len(df) > 19999:
        df.loc[19999, ["comment_number", "comment"]] = [4, "A dog runs across the grass."]
    df["comment_number"] = pd.to_numeric(df["comment_number"], errors="coerce")

    unique_images = df["image_name"].unique()
    if cap is not None:
        cap_images = min(cap // 5, len(unique_images))
        unique_images = unique_images[:cap_images]
        df = df[df["image_name"].isin(unique_images)].copy()
        print(f"  Cap applied : {cap_images:,} images -> {len(df):,} rows")

    # ── Deterministic shuffle on image IDs only ──────────────────────
    rng = np.random.default_rng(seed)
    unique_images = df["image_name"].unique().copy()
    rng.shuffle(unique_images)

    n_total = len(unique_images)
    n_train = math.floor(n_total * train_split)
    n_val = math.floor(n_total * val_split)
    # test gets whatever remains -- guarantees no rounding gap

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train:n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val:])

    assert len(train_imgs & val_imgs) == 0
    assert len(train_imgs & test_imgs) == 0
    assert len(val_imgs & test_imgs) == 0
    assert len(train_imgs) + len(val_imgs) + len(test_imgs) == n_total

    def assign_split(image_name):
        if image_name in train_imgs:
            return "train"
        if image_name in val_imgs:
            return "val"
        return "test"

    df["split"] = df["image_name"].apply(assign_split)

    df.to_csv(split_csv_path, index=False)
    if on_artifact_saved:
        on_artifact_saved(split_csv_path, "flickr30_split.csv")

    print(f"  Split CSV saved -> {split_csv_path}")
    print(f"  Images  : train={len(train_imgs):,}  val={len(val_imgs):,}  test={len(test_imgs):,}")
    print(f"  Rows    : train={df[df.split == 'train'].shape[0]:,}  "
          f"val={df[df.split == 'val'].shape[0]:,}  "
          f"test={df[df.split == 'test'].shape[0]:,}")

    # ── Build vocab from ALL captions (before any split filtering) ─────
    # Must use all rows so val/test words are in the vocab too.
    spacy_model = spacy.load("en_core_web_sm")

    def tokenize(caption):
        return [f" {tok.text.lower()}" for tok in spacy_model(str(caption))]

    counter = Counter()
    for caption in df["comment"].tolist():
        counter.update(tokenize(caption))

    stoi = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3, "<EN>": 4}
    idx = 5
    stoi.update({s: i + idx for i, (s, _) in enumerate(counter.most_common(10000))})
    current_idx = max(stoi.values())
    for ch in string.ascii_lowercase + " ":
        if ch not in stoi:
            current_idx += 1
            stoi[ch] = current_idx
    itos = {i: s for s, i in stoi.items()}

    with open(vocab_stoi_path, "wb") as f:
        pickle.dump(stoi, f)
    with open(vocab_itos_path, "wb") as f:
        pickle.dump(itos, f)

    print(f"  Vocab saved -> {vocab_stoi_path}  ({len(stoi):,} tokens)")
    if on_artifact_saved:
        on_artifact_saved(vocab_stoi_path, "vocab_stoi.pkl")
        on_artifact_saved(vocab_itos_path, "vocab_itos.pkl")
    print("[prepare_data] Done.\n")
