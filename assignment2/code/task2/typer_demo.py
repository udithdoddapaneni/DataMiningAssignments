# GiG


from typing import Annotated

import typer

from ds5612_pa2.code import pipeline_configs


# Typer has some interesting feature that allows you to
# add some completion options to your shell
# This is useful in a large setting but a distraction for toy scripts
# So disable it.
app = typer.Typer(add_completion=False)


@app.command()
def train(
    classifier: Annotated[
        pipeline_configs.ValidClassifierNames, typer.Option(help="Classifier Name")
    ] = "DecisionTree",
) -> None:
    """Train a classifier."""
    pipeline = pipeline_configs.get_simple_ml_pipeline(classifier)
    pipeline.run_pipeline()


@app.command()
def predict(
    item: list[float],
    classifier: Annotated[
        pipeline_configs.ValidClassifierNames, typer.Option(help="Classifier Name")
    ] = "DecisionTree",
) -> None:
    """Make a prediction using a trained classifier"""
    print(pipeline_configs.get_prediction_probabilities(item, classifier))


if __name__ == "__main__":
    app()
