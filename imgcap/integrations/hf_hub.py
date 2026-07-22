"""Thin optional Hugging Face Hub wrapper for checkpoint/artifact backup.

Everything here is opt-in via `cfg.use_hf_hub`. Requires `HF_TOKEN` in the
environment (or being already logged in via `huggingface-cli login`).
"""
import os


class HFHubUploader:
    def __init__(self, repo_id: str, private: bool = False):
        from huggingface_hub import create_repo, login, upload_file

        self._upload_file = upload_file
        self.repo_id = repo_id

        token = os.environ.get("HF_TOKEN")
        if token:
            login(token)

        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print("HF Hub repo:", repo_id)

    def upload(self, local_path: str, path_in_repo: str) -> None:
        self._upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
        )
        print(f"  \u2713 Uploaded {local_path} -> {self.repo_id}/{path_in_repo}")


def build_uploader(cfg):
    """Returns an HFHubUploader if cfg.use_hf_hub, else None."""
    if not cfg.use_hf_hub:
        return None
    if not cfg.hf_repo_id:
        raise ValueError("cfg.use_hf_hub=True requires cfg.hf_repo_id to be set.")
    return HFHubUploader(cfg.hf_repo_id, private=cfg.hf_private_repo)


def download_file(repo_id: str, filename: str) -> str:
    """Convenience wrapper around hf_hub_download, for pulling pretrained
    checkpoints / prepared vocab & split files from the Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)
