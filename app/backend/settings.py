"""Settings of the application. Nothing here is stored in the repository: the database, the weights and the
uploaded images live in a data folder of the user (MRI_APP_DATA, default ~/.mri-classifier)."""
import os
from dataclasses import dataclass, field


def default_data_dir():
    return os.environ.get("MRI_APP_DATA") or os.path.join(os.path.expanduser("~"), ".mri-classifier")


@dataclass
class Settings:
    data_dir: str = field(default_factory=default_data_dir)
    retrain_min: int = 20           # reviewed images needed before a retraining round
    margin_threshold: float = 0.80  # below this margin an image counts as uncertain (results/margin_threshold.txt: validation, seeds 42-44)
    blocked_manifest: str = ""      # split manifest (tsv with md5 and split columns); its test and external images never train
    blocked_splits: tuple = ("test", "external")
    ft_epochs: int = 5              # fine-tuning on the reviewed images of one round
    ft_lr: float = 1e-5             # small, so the model does not forget what best.pt learned
    max_forget_drop: float = 0.05   # a new model is not promoted if accuracy on earlier rounds' images drops more than this

    @property
    def db_path(self):
        return os.path.join(self.data_dir, "app.sqlite")

    @property
    def weights_dir(self):
        return os.path.join(self.data_dir, "models")

    @property
    def images_dir(self):
        return os.path.join(self.data_dir, "images")

    def ensure_dirs(self):
        for d in (self.data_dir, self.weights_dir, self.images_dir):
            os.makedirs(d, exist_ok=True)
