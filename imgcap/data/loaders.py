from torch.utils.data import DataLoader
from torchvision import transforms as T

from imgcap.config import TrainConfig
from imgcap.data.dataset import BucketSampler, Flickr30, collate_fn
from imgcap.data.prepare import prepare_data

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomApply([
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.RandomCrop(size=(110, 110)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.8),
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_loaders(cfg: TrainConfig, on_artifact_saved=None):
    """Returns (train_loader, val_loader, test_loader, vocab_size, pad_idx, train_ds)."""

    prepare_data(
        captions_csv=cfg.captions_csv,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        seed=cfg.seed,
        split_csv_path=cfg.split_csv_path,
        vocab_stoi_path=cfg.vocab_stoi_path,
        vocab_itos_path=cfg.vocab_itos_path,
        cap=cfg.cap,
        on_artifact_saved=on_artifact_saved,
    )

    train_transform, eval_transform = build_transforms()

    print("Loading datasets ...")
    train_ds = Flickr30(cfg.images_dir, cfg.split_csv_path, cfg.vocab_stoi_path,
                         cfg.vocab_itos_path, split="train", transform=train_transform)
    val_ds = Flickr30(cfg.images_dir, cfg.split_csv_path, cfg.vocab_stoi_path,
                       cfg.vocab_itos_path, split="val", transform=eval_transform)
    test_ds = Flickr30(cfg.images_dir, cfg.split_csv_path, cfg.vocab_stoi_path,
                        cfg.vocab_itos_path, split="test", transform=eval_transform)

    vocab_size = len(train_ds.vocab_stoi)
    pad_idx = train_ds.vocab_stoi["<pad>"]
    print(f"Vocabulary size : {vocab_size:,}")

    train_sampler = BucketSampler(train_ds.caption_lengths, cfg.batch_size,
                                   bucket_size=100, shuffle=True)
    val_sampler = BucketSampler(val_ds.caption_lengths, cfg.batch_size,
                                 bucket_size=100, shuffle=False)
    test_sampler = BucketSampler(test_ds.caption_lengths, cfg.batch_size,
                                  bucket_size=100, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                               num_workers=cfg.num_workers, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                             num_workers=cfg.num_workers, pin_memory=True,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, sampler=test_sampler,
                              num_workers=cfg.num_workers, pin_memory=True,
                              collate_fn=collate_fn)

    print(f"Train batches   : {len(train_loader):,}")
    print(f"Val   batches   : {len(val_loader):,}")
    print(f"Test  batches   : {len(test_loader):,}")

    return train_loader, val_loader, test_loader, vocab_size, pad_idx, train_ds
