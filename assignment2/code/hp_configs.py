# GiG
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.panel import Panel
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)

from ds5612_pa2.code import pipeline_configs, utils


VALID_HP_TUNING_TYPES = (
    GridSearchCV | HalvingGridSearchCV | RandomizedSearchCV | HalvingRandomSearchCV
)
assert enable_halving_search_cv is not None


console = Console()


class HyperParameterTuningConfig(BaseModel, ABC):
    """HyperParameterConfig is an abstract class at root of all HP tuners."""

    # Do not allow any new fields when creating this class.
    # Pydantic freaks out when you use NDArray type annotation
    # So use the arbitrary_types_allowed config.

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    scoring: utils.MODEL_METRIC = "accuracy"

    @abstractmethod
    def create_hyperparameter_tuner(
        self, ml_model_config: pipeline_configs.MLModelConfig
    ) -> VALID_HP_TUNING_TYPES:
        """create_hyperparameter_tuner creates appropriate hyper parameter tuner object."""


##################Start modifying here##################
class GridSearchConfig(HyperParameterTuningConfig):
    """GridSearchConfig is a class to store appropriate config for GridSearchCV."""

    successive_halving: bool = False

    def create_hyperparameter_tuner(
        self, ml_model_config: pipeline_configs.MLModelConfig
    ) -> GridSearchCV | HalvingGridSearchCV:
        """create_hyperparameter_tuner creates appropriate hyper parameter tuner object."""

        estimator = ml_model_config.create_classifier()
        param_grid = ml_model_config.get_hyper_param_grid()
        return GridSearchCV(estimator, param_grid)


class RandomizedSearchConfig(HyperParameterTuningConfig):
    """RandomizedSearchConfig is a class to store appropriate config for RandomizedSearchCV."""

    successive_halving: bool = False

    def create_hyperparameter_tuner(
        self, ml_model_config: pipeline_configs.MLModelConfig
    ) -> RandomizedSearchCV | HalvingRandomSearchCV:
        """create_hyperparameter_tuner creates appropriate hyper parameter tuner object."""

        estimator = ml_model_config.create_classifier()
        param_grid = ml_model_config.get_hyper_param_grid()
        return RandomizedSearchCV(estimator, param_grid)


##################End modifying here##################


########################Begin: Do not modify anything below########################
class MLPipelineWithHPTuning(pipeline_configs.SimpleMLPipeline):
    """MLPipelineWithHPTuning creates a ML pipeline that can use HP tuning."""

    hp_config: HyperParameterTuningConfig
    hp_tuner: VALID_HP_TUNING_TYPES | None = None

    def tune_hyper_parameters(self) -> None:
        """tune_hyper_parameters uses the appropriate algorithms to do HP tuning."""
        self.hp_tuner = self.hp_config.create_hyperparameter_tuner(self.ml_model_config)
        self.hp_tuner.fit(self.dataset.train_and_validation_X, self.dataset.train_and_validation_y)

    def print_evaluation_results(self, verbose: bool = True) -> dict:  # noqa: FBT001, FBT002
        """Evaluate is a dummy function that prints the accuracy of a classifier."""
        assert self.hp_tuner is not None
        pred_y = self.hp_tuner.predict(self.dataset.test_X)
        name = self.ml_model_config.get_classifier_name()
        if verbose:
            console.print(f"[bold red]Classification report for {name}: [/bold red]")
            console.print(classification_report(self.dataset.test_y, pred_y))
        report = classification_report(self.dataset.test_y, pred_y, output_dict=True)
        assert isinstance(report, dict)
        return report

    def run_pipeline(self, verbose: bool = True) -> None:  # noqa: FBT001, FBT002
        """run_pipeline runs the simple ML pipeline."""
        if verbose:
            console.print(
                Panel("[bold blue]Without hyper parameter tuning.[/bold blue]", expand=False)
            )
        self.train()
        if verbose:
            super().print_evaluation_results()
            console.print(
                Panel("[bold blue]With hyper parameter tuning.[/bold blue]", expand=False)
            )
        self.tune_hyper_parameters()
        self.print_evaluation_results(verbose)


class HyperParameterAlgos(str, Enum):
    """ValidClassifierNames is an Enum to control which classifiers are used in ML pipeline."""

    GridSearch = "GridSearch"
    RandomizedSearch = "RandomizedSearch"
    HalvingGridSearchCV = "HalvingGridSearchCV"
    HalvingRandomSearch = "HalvingRandomSearch"

    # By default, Enums output the "name" (eg DT, NB and KNN)
    # Make it output the value
    def __str__(self) -> str:
        """Customize the display name of the enum variable."""
        return str(self.value)


def get_hp_config(hp_name: str) -> HyperParameterTuningConfig:
    """get_classifier_config returns a model config object with default params."""
    match hp_name:
        case HyperParameterAlgos.GridSearch.name | HyperParameterAlgos.GridSearch.value:
            return GridSearchConfig()
        case HyperParameterAlgos.RandomizedSearch.name | HyperParameterAlgos.RandomizedSearch.value:
            return RandomizedSearchConfig()
        case (
            HyperParameterAlgos.HalvingGridSearchCV.name
            | HyperParameterAlgos.HalvingGridSearchCV.value
        ):
            return GridSearchConfig(successive_halving=True)
        case (
            HyperParameterAlgos.HalvingRandomSearch.name
            | HyperParameterAlgos.HalvingRandomSearch.value
        ):
            return RandomizedSearchConfig(successive_halving=True)
        case _:
            raise ValueError(f"Invalid hyper parameter tuning algorithm: {hp_name}")


def get_hp_ml_pipeline(
    classifier_name: str = "DT", hp_name: str = "RandomSearch"
) -> MLPipelineWithHPTuning:
    """get_simple_ml_pipeline produces a pipeline object with default params."""
    d = pipeline_configs.DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    ml_model_config = pipeline_configs.get_classifier_config(classifier_name)
    return MLPipelineWithHPTuning(
        dataset=d, ml_model_config=ml_model_config, hp_config=get_hp_config(hp_name)
    )


########################End: Do not modify anything below########################

if __name__ == "__main__":
    get_hp_ml_pipeline("DT", "RandomizedSearch").run_pipeline()
    get_hp_ml_pipeline("DT", "GridSearch").run_pipeline()
    get_hp_ml_pipeline("DT", "HalvingGridSearchCV").run_pipeline()

    # Since our HP tuning is not that good for KNN and NB,
    # no need to try them :)
    # get_hp_ml_pipeline("KNN", "RandomizedSearch").run_pipeline()
    # get_hp_ml_pipeline("NB", "RandomizedSearch").run_pipeline()
