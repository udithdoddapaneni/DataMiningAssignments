# GiG

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification


NUM_FEATURES = 10
NUM_TUPLES = 20000

ANSWER_TO_EVERYTHING = 42
MODEL_METRIC = Literal["accuracy", "precision", "recall", "f1"]


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """get_project_root returns the absolute path to the project root directory.

    The result is cached, so subsequent calls return immediately.
    """
    # Get the path of the current file
    current_file = Path(__file__).resolve()

    # Navigate up until we find the pyproject.toml file
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # If we couldn't find the root, raise an exception
    raise FileNotFoundError("Couldn't find the project root directory.")


def create_dataset(
    num_tuples: int = NUM_TUPLES, num_features: int = NUM_FEATURES, random_state: int = 42
) -> NDArray[np.float64]:
    """create_dataset creates a toy synthetic dataset."""
    X, y = make_classification(
        random_state=random_state,
        n_samples=num_tuples,
        n_features=num_features,
        n_informative=int(0.6 * num_features),
        n_redundant=int(0.4 * num_features),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        shuffle=True,
    )

    # Combine the data for ease of use
    return np.column_stack((X, y))


def save_dataset(X_y: NDArray[np.float64], file_path: Path) -> None:  # noqa: N803
    """save_dataset stores the data to a file with a hard coded name."""
    np.savez_compressed(file_path, data=X_y)
    print(f"Saved to file: {file_path}")


def load_dataset(file_path: Path) -> NDArray[np.float64]:
    """load_dataset loads a dataset from a hard coded file name."""
    loaded_data = np.load(file_path)
    return loaded_data["data"]


DATASET_FILE_PATH = get_project_root() / "src/ds5612_pa2/resources/data_Xy.npz"
