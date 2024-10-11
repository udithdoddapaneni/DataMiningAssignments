# GiG

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
from rich.console import Console
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ds5612_pa2.code import utils


console = Console()


VALID_CLASSIFIER_TYPES = DecisionTreeClassifier | KNeighborsClassifier | GaussianNB


################## Begin: Do not change anything below##################


class ValidClassifierNames(str, Enum):
    """ValidClassifierNames is an Enum to control which classifiers are used in ML pipeline."""

    DT = "DecisionTree"
    NB = "NaiveBayes"
    KNN = "KNN"

    # By default, Enums output the "name" (eg DT, NB and KNN)
    # Make it output the value
    def __str__(self) -> str:
        """Customize the display name of the enum variable."""
        return str(self.value)


class MLModelConfig(BaseModel, ABC):
    """MLModelConfigBaseClass is an abstract base class."""

    # Do not allow any new fields when creating this class.
    # Pydantic freaks out when you use NDArray type annotation
    # So use the arbitrary_types_allowed config.

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @abstractmethod
    def create_classifier(self) -> VALID_CLASSIFIER_TYPES:
        """create_classifier is an abstract function that can create appropriate classifiers."""

    @abstractmethod
    def get_classifier_name(self) -> ValidClassifierNames:
        """get_classifier_name outputs the name of the classifier."""

    @abstractmethod
    def get_hyper_param_grid(self) -> dict[str, list]:
        """get_hyper_param_grid outputs dict using for HP tuning."""


################## End: Do not change anything below##################


##################Start modifying here##################
class DatasetConfig(BaseModel):
    """DatasetConfig class is used to represent a dataset."""

    # Do not allow any new fields when creating this class.
    # Pydantic freaks out when you use NDArray type annotation
    # So use the arbitrary_types_allowed config.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    file_path: FilePath = Field(description="Path to the dataset file")
    train_size: float = Field(
        default=0.3, ge=0.1, le=0.5, description="Proportion of data to use for training"
    )
    validation_size: float = Field(
        default=0.1, ge=0.1, le=0.2, description="Proportion of data to use for validation"
    )
    test_size: float = Field(
        default=0.1, ge=0.1, le=0.3, description="Proportion of data to use for testing"
    )
    production_size: float = Field(
        default=0.5, ge=0.1, le=0.5, description="Proportion of data to use for Production"
    )

    @model_validator(mode="after")
    def check_sum(self):
        if self.train_size + self.validation_size + self.test_size + self.production_size != 1:
            raise ValueError(
                "Fields train_size, validation_size, test_size, production_size should add up to 1."
            )
        return self

    ################## Begin: Do not change anything below##################

    full_data: NDArray[np.float64] = np.array([])
    train_X: NDArray[np.float64] = np.array([])  # noqa: N815
    validation_X: NDArray[np.float64] = np.array([])  # noqa: N815
    train_and_validation_X: NDArray[np.float64] = np.array([])  # noqa: N815
    test_X: NDArray[np.float64] = np.array([])  # noqa: N815
    train_val_test_X: NDArray[np.float64] = np.array([])  # noqa: N815
    production_X: NDArray[np.float64] = np.array([])  # noqa: N815

    train_y: NDArray[np.float64] = np.array([])
    validation_y: NDArray[np.float64] = np.array([])
    train_and_validation_y: NDArray[np.float64] = np.array([])
    test_y: NDArray[np.float64] = np.array([])
    train_val_test_y: NDArray[np.float64] = np.array([])
    production_y: NDArray[np.float64] = np.array([])

    def load_data(self) -> None:
        """load_data reads a npz file and splits the dataset based on the configs."""
        self.full_data = utils.load_dataset(self.file_path)

        num_tuples = len(self.full_data)

        # We will split X and y into 4 parts.
        # If the config is 0.3, 0.1, 0.1 and 0.5, then
        # for 20K, the splits will be 0-6K, 6001-8K, 8001-10K, 10001-20K

        index_start, index_end = 0, int(self.train_size * num_tuples)
        temp_Xy = self.full_data[0:index_end]
        self.train_X, self.train_y = temp_Xy[:, :-1], temp_Xy[:, -1]

        index_start, index_end = index_end, index_end + int(self.validation_size * num_tuples)
        temp_Xy = self.full_data[index_start:index_end]
        self.validation_X, self.validation_y = temp_Xy[:, :-1], temp_Xy[:, -1]

        self.train_and_validation_X = np.vstack((self.train_X, self.validation_X))
        self.train_and_validation_y = np.hstack((self.train_y, self.validation_y))

        index_start, index_end = index_end, index_end + int(self.test_size * num_tuples)
        temp_Xy = self.full_data[index_start:index_end]
        self.test_X, self.test_y = temp_Xy[:, :-1], temp_Xy[:, -1]

        temp_Xy = self.full_data[0:index_end]
        self.train_val_test_X, self.train_val_test_y = temp_Xy[:, :-1], temp_Xy[:, -1]

        index_start, index_end = index_end, num_tuples
        temp_Xy = self.full_data[index_start:index_end]
        self.production_X, self.production_y = temp_Xy[:, :-1], temp_Xy[:, -1]

    def print_dataset_stats(self) -> None:
        """print_stats just prints some basic statistics."""
        console.print(f"Train: {self.train_X.shape}, {self.train_y.shape}")
        console.print(f"Validation: {self.validation_X.shape}, {self.validation_y.shape}")
        console.print(
            f"Train+Val: {self.train_and_validation_X.shape}, {self.train_and_validation_y.shape}"
        )
        console.print(f"Test: {self.test_X.shape}, {self.test_y.shape}")
        console.print(
            f"Train+Val+Test: {self.train_val_test_X.shape}, {self.train_val_test_y.shape}"
        )
        console.print(f"Production: {self.production_X.shape}, {self.production_y.shape}")


################## End: Do not change anything below##################


class DecisionTreeConfig(MLModelConfig):
    """DecisionTreeConfig specifies params to create DecisionTreeClassifier."""

    ml_model_type: Literal["decision_tree"] = "decision_tree"
    random_state: int = Field(default=utils.ANSWER_TO_EVERYTHING)

    criterion: Literal["gini", "entropy", "log_loss"] = Field(
        default="gini", description="Function to measure the quality of a split"
    )
    max_depth: int = Field(default=1, ge=1, description="Maximum depth of the tree")
    min_samples_split: int = Field(
        default=2, ge=2, description="Minimum number of samples required to split an internal node"
    )

    ################## Begin: Do not change anything below##################
    def create_classifier(self) -> DecisionTreeClassifier:
        """create_classifier returns a DecisionTreeClassifier that is appropriately initialized."""
        return DecisionTreeClassifier(
            random_state=self.random_state,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )

    def get_classifier_name(self) -> ValidClassifierNames:
        """get_classifier_name outputs the name of the classifier."""
        return ValidClassifierNames.DT

    def get_hyper_param_grid(self) -> dict[str, list]:
        """get_hyper_param_grid outputs dict using for HP tuning."""
        return {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [4, 8, 16],
            "min_samples_split": [4, 8, 16],
        }


################## End: Do not change anything below##################


class NaiveBayesConfig(MLModelConfig):
    """NaiveBayesConfig specifies params to create naive_bayes."""

    ml_model_type: Literal["naive_bayes"] = "naive_bayes"
    variant: Literal["gaussian"] = Field(
        default="gaussian", description="Variant of Naive Bayes to use"
    )

    ################## Begin: Do not change anything below##################
    def create_classifier(self) -> GaussianNB:
        """create_classifier returns a DecisionTreeClassifier that is appropriately initialized."""
        return GaussianNB()

    def get_classifier_name(self) -> ValidClassifierNames:
        """get_classifier_name outputs the name of the classifier."""
        return ValidClassifierNames.NB

    def get_hyper_param_grid(self) -> dict[str, list]:
        """get_hyper_param_grid outputs dict using for HP tuning."""
        return {
            "var_smoothing": [1e-9, 1e-8, 1e-4],
        }

    ################## End: Do not change anything below##################


class KNNConfig(MLModelConfig):
    """KNNConfig specifies params to create KNeighborsClassifier."""

    ml_model_type: Literal["knn"] = "knn"
    n_neighbors: int = Field(default=5, ge=1, description="Number of neighbors to use")
    weights: Literal["uniform", "distance"] = Field(
        default="uniform", description="Weight function used in prediction"
    )
    p: int = Field(default=2, ge=1, le=3, description="Power parameter for the Minkowski metric")

    ################## Begin: Do not change anything below##################
    def create_classifier(self) -> KNeighborsClassifier:
        """create_classifier returns a DecisionTreeClassifier that is appropriately initialized."""
        return KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)

    def get_classifier_name(self) -> ValidClassifierNames:
        """get_classifier_name outputs the name of the classifier."""
        return ValidClassifierNames.KNN

    def get_hyper_param_grid(self) -> dict[str, list]:
        """get_hyper_param_grid outputs dict using for HP tuning."""
        return {
            "n_neighbors": [1, 2, 4, 8, 16],
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3, 4],
        }


################## End: Do not change anything below##################


########################Begin: Do not modify anything below########################
class SimpleMLPipeline(BaseModel):
    """SimpleMLPipeline is an abstraction of ML pipeline to build and evaluate a classifier."""

    # Do not allow any new fields when creating this class.
    # Pydantic freaks out when you use NDArray type annotation
    # So use the arbitrary_types_allowed config.

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(default="", description="Name of the ML pipeline")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of pipeline creation"
    )

    dataset: DatasetConfig
    ml_model_config: MLModelConfig

    classifier: VALID_CLASSIFIER_TYPES | None = None

    def get_prediction_class(self, features: list[float]) -> int:
        """make_prediction takes a single item and outputs the prediction probabilities."""
        assert self.classifier is not None
        single_item = np.array(features).reshape(1, -1)
        prediction = self.classifier.predict(single_item)
        return int(prediction[0])

    def get_prediction_probabilities(self, features: list[float]) -> tuple[float, float]:
        """make_prediction takes a single item and outputs the prediction probabilities."""
        assert self.classifier is not None
        single_item = np.array(features).reshape(1, -1)
        prediction_proba = self.classifier.predict_proba(single_item)
        return tuple(prediction_proba[0].tolist())

    def print_evaluation_results(self, verbose: bool = True) -> dict:  # noqa: FBT001, FBT002
        """Evaluate is a dummy function that prints the accuracy of a classifier."""
        assert self.classifier is not None
        pred_y = self.classifier.predict(self.dataset.test_X)
        name = self.ml_model_config.get_classifier_name()
        if verbose:
            console.print(f"[bold red]Classification report for {name}: [/bold red]")
            console.print(classification_report(self.dataset.test_y, pred_y))
        report = classification_report(self.dataset.test_y, pred_y, output_dict=True)
        assert isinstance(report, dict)
        return report

    def train(self) -> None:
        """Train creates a classifier and fits it to data."""
        self.dataset.load_data()
        self.classifier = self.ml_model_config.create_classifier()
        self.classifier.fit(self.dataset.train_X, self.dataset.train_y)

    def run_pipeline(self, verbose: bool = True) -> None:  # noqa: FBT001, FBT002
        """run_pipeline runs the simple ML pipeline."""
        self.train()
        self.print_evaluation_results(verbose)


# Advanced: Okay to ignore.
# Two interesting things to note here:
#  1. First, we are using | to test multiple values in single match case.
#  2. We are using the name and value params of an Enum to test both cases
def get_classifier_config(classifier_name: str) -> MLModelConfig:
    """get_classifier_config returns a model config object with default params."""
    match classifier_name:
        case ValidClassifierNames.DT.name | ValidClassifierNames.DT.value:
            return DecisionTreeConfig()
        case ValidClassifierNames.KNN.name | ValidClassifierNames.KNN.value:
            return KNNConfig()
        case ValidClassifierNames.NB.name | ValidClassifierNames.NB.value:
            return NaiveBayesConfig()
        case _:
            raise ValueError(f"Invalid classifier name {classifier_name}")


def get_simple_ml_pipeline(classifier_name: str = "DT") -> SimpleMLPipeline:
    """get_simple_ml_pipeline produces a pipeline object with default params."""
    d = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    ml_model_config = get_classifier_config(classifier_name)
    return SimpleMLPipeline(dataset=d, ml_model_config=ml_model_config)


def get_prediction_probabilities(
    features: list[float],
    classifier: str,
) -> tuple[float, float]:
    """get_prediction_probabilities creates a pipeline and returns probabilities."""
    pipeline = get_simple_ml_pipeline(classifier)
    pipeline.train()
    return pipeline.get_prediction_probabilities(features)


def get_prediction_class(
    features: list[float],
    classifier: str,
) -> int:
    """get_prediction_probabilities creates a pipeline and returns probabilities."""
    pipeline = get_simple_ml_pipeline(classifier)
    pipeline.train()
    return pipeline.get_prediction_class(features)


########################End: Do not modify anything below########################

if __name__ == "__main__":
    get_simple_ml_pipeline("DT").run_pipeline()
    get_simple_ml_pipeline("KNN").run_pipeline()
    get_simple_ml_pipeline("NB").run_pipeline()
