"""Download and prepare the public TrashNet image dataset."""

from pathlib import Path
import shutil
from urllib.request import urlretrieve
from zipfile import ZipFile


PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = PROJECT_DIR / "trashnet_dataset_resized.zip"
IMAGE_DATASET_DIR = PROJECT_DIR / "image_dataset"
DATASET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/"
    "resolve/main/dataset-resized.zip"
)


def download_and_prepare():
    """Download 2,527 labelled images and extract folder-per-class data."""
    if not ARCHIVE_PATH.exists():
        print("Downloading the public TrashNet dataset (~42 MB)...")
        urlretrieve(DATASET_URL, ARCHIVE_PATH)

    IMAGE_DATASET_DIR.mkdir(exist_ok=True)
    with ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(PROJECT_DIR / "_trashnet_extract")

    extracted_root = PROJECT_DIR / "_trashnet_extract" / "dataset-resized"
    if not extracted_root.exists():
        raise FileNotFoundError("The downloaded archive has an unexpected layout.")

    for class_dir in extracted_root.iterdir():
        if class_dir.is_dir():
            destination = IMAGE_DATASET_DIR / class_dir.name.title()
            destination.mkdir(exist_ok=True)
            for image_path in class_dir.iterdir():
                target = destination / image_path.name
                if not target.exists():
                    image_path.replace(target)

    shutil.rmtree(PROJECT_DIR / "_trashnet_extract")
    print(f"Prepared image folders at: {IMAGE_DATASET_DIR}")
    print("Classes: cardboard, glass, metal, paper, plastic, trash")


if __name__ == "__main__":
    download_and_prepare()
