# GiG

from typing import Any

from rich.console import Console
from rich.panel import Panel
from sklearn.metrics import classification_report

from ds5612_pa2.code import utils
from ds5612_pa2.code.error_estimation import ErrorEstimationAlgos, ErrorEstimator, estimate_error
from ds5612_pa2.code.hp_configs import (
    HyperParameterTuningConfig,
    get_hp_config,
)
from ds5612_pa2.code.pipeline_configs import (
    DatasetConfig,
    MLModelConfig,
    get_classifier_config,
)


console = Console()


class ModernMLPipeline:
    """ModernMLPipeline demonstrates how a modern ML pipeline works."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        ml_model_config: MLModelConfig,
        hp_config: HyperParameterTuningConfig,
    ) -> None:
        self.dataset_config = dataset_config
        self.ml_model_config = ml_model_config
        self.hp_config = hp_config

    def get_production_score(self, classifier: Any, title: str, verbose=True) -> dict:  # noqa: ANN001, ANN401, FBT002
        """get_production_score returns the classification report as dict."""
        pred_y = classifier.predict(self.dataset_config.production_X)
        if verbose:
            console.print(Panel(f"[bold blue]{title}[/bold blue]", expand=False))
            console.print(classification_report(self.dataset_config.production_y, pred_y))
        report_dict = classification_report(
            self.dataset_config.production_y, pred_y, output_dict=True
        )
        assert isinstance(report_dict, dict)
        return report_dict

    ######################Begin: Make changes here######################
    def step_1(self) -> None:
        """step_1 loads the data and trains a classifier on training data."""
        self.dataset_config.load_data()
        self.step1_classifier = self.ml_model_config.create_classifier()
        self.step1_classifier.fit(self.dataset_config.train_X, self.dataset_config.train_y)

    def step_2_3_4(self) -> None:
        """step_2_3_4 does HP tuning and trains a classifier with best HP values."""
        tuner = self.hp_config.create_hyperparameter_tuner(ml_model_config=self.ml_model_config)
        tuner.fit(self.dataset_config.train_and_validation_X, self.dataset_config.train_and_validation_y)
        self.step2_classifier = tuner.best_estimator_

    def step_5(self) -> None:
        """step_5 does error estimation."""
        self.error_estimator = estimate_error(
            dataset_config=self.dataset_config,
            ml_model=self.ml_model_config.create_classifier(),
            error_estimator=ErrorEstimator(algorithm=ErrorEstimationAlgos(value=ErrorEstimationAlgos.TRAIN_VAL))
        )

    def step_6(self) -> None:
        """step_6 trains model on full data."""
        self.step6_classifier = self.step2_classifier
        self.step6_classifier.fit(self.dataset_config.train_val_test_X, self.dataset_config.train_val_test_y)

    ######################End: Make changes here######################

    def run_pipeline(self, verbose: bool = True) -> None:  # noqa: FBT001, FBT002
        """run_pipeline runs the full ML pipeline."""
        self.step_1()
        self.get_production_score(self.step1_classifier, "Classifier: Train", verbose=verbose)
        self.step_2_3_4()
        self.get_production_score(
            self.step2_classifier, "HP-Classifier: Train+Val", verbose=verbose
        )
        self.step_5()
        self.step_6()
        self.get_production_score(
            self.step6_classifier, "HP-Classifier: Train+Val+Test", verbose=verbose
        )


def get_full_ml_pipeline(
    classifier_name: str = "DT", hp_name: str = "RandomSearch"
) -> ModernMLPipeline:
    """get_simple_ml_pipeline produces a pipeline object with default params."""
    d = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    ml_model_config = get_classifier_config(classifier_name)
    return ModernMLPipeline(
        dataset_config=d, ml_model_config=ml_model_config, hp_config=get_hp_config(hp_name)
    )


if __name__ == "__main__":
    get_full_ml_pipeline("DT", "RandomizedSearch").run_pipeline()
