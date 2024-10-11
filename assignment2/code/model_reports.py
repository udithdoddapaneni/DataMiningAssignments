# GiG

from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sklearn import metrics
from sklearn.metrics import classification_report

from ds5612_pa2.code import utils
from ds5612_pa2.code.pipeline_configs import (
    DatasetConfig,
    get_classifier_config,
)


def generate_report_data() -> list:
    dataset = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    dataset.load_data()
    classifier_report_data = []
    for classifier in ["DT", "KNN", "NB"]:
        ml_model = get_classifier_config(classifier).create_classifier()
        ml_model.fit(dataset.train_val_test_X, dataset.train_val_test_y)
        pred_y = ml_model.predict(dataset.production_X)
        report = classification_report(dataset.production_y, pred_y, output_dict=True)

        report["accuracy"] = {"accuracy":report["accuracy"], "support": 0}
        for label in report:
            if label != "accuracy":
                report[label]["precision"] = report[label]["precision"]
                report[label]["recall"] = report[label]["recall"]
                report[label]["f1-score"] = report[label]["f1-score"]
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                report["accuracy"]["support"] += report[label]["support"]

        fpr, tpr, _ = metrics.roc_curve(dataset.production_y, pred_y)

        report_data = {
            "classifier_name": classifier,
            "classification_report": report,
            "fpr_tpr": [{"x": fpr[i], "y": tpr[i]} for i in range(len(fpr))],
        }
        classifier_report_data.append(report_data)
    return classifier_report_data


def render_report() -> None:
    classifier_report_data = generate_report_data()

    base_template_path = utils.get_project_root() / "src/ds5612_pa2/code/"
    file_loader = FileSystemLoader(base_template_path)
    env = Environment(loader=file_loader, autoescape=True)
    template = env.get_template("model_report_template.jinja2")

    with Path.open(base_template_path / "output.html", "w") as f:
        print(template.render(classifiers=classifier_report_data), file=f)


if __name__ == "__main__":
    render_report()
