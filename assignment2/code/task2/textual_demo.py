# GiG

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, ProgressBar, Select, TextArea

from ds5612_pa2.code import pipeline_configs


CLASSIFIER_OPTIONS = [(clf.value, clf.name) for clf in pipeline_configs.ValidClassifierNames]


def predict(item_str: str, classifier: str) -> tuple[float, float]:
    """Predict using the specified classifier and input values."""
    item = [float(elem) for elem in item_str.split()]
    result = pipeline_configs.get_prediction_probabilities(item, classifier)
    return result[0], result[1]


class TextualClassifier(App):
    """TextualClassifier is a simple demo for a ML classifier."""

    CSS = """
    Horizontal {
        height: 100%;
    }
    #input-panel, #output-panel {
        width: 50%;
        height: 100%;
        padding: 1;
    }
    #input-panel {
        border-right: solid green;
    }
    TextArea {
        height: 1fr;
    }
    ProgressBar {
        margin: 1 0;
    }
    #predict {
        margin-top: 1;
    }
    Label {
        padding-top: 1;
        padding-bottom: 1;
    }
    Select {
        width: 60;
    }
    """
    positive_progress = 0
    negative_progress = 0

    def compose(self) -> ComposeResult:
        inputbox = Vertical(
            Label("Classifier:"),
            Select(id="classifier", options=CLASSIFIER_OPTIONS),
            Label("Input Text:"),
            TextArea(id="input-text"),
            Button(label="predict", id="predict"),
            id="input-panel",
        )
        # positive_progress = self.query_one("positive", ProgressBar).
        outputbox = Vertical(
            Label("Classifier Ouput"),
            Label(f"Positive Score: {self.positive_progress}", id="positive_score"),
            ProgressBar(total=1, id="positive"),
            Label(f"Negative Score: {self.negative_progress}", id="negative_score"),
            ProgressBar(total=1, id="negative"),
            id="output-panel",
        )
        final_box = Horizontal(inputbox, outputbox)
        """Compose creates the UI."""
        yield Header()
        yield final_box
        yield Footer()

    def on_mount(self) -> None:
        """on_mount can be used to set initial values."""
        # Set default values for progress bars
        self.query_one("#positive", ProgressBar).update(progress=0, total=1)
        self.query_one("#negative", ProgressBar).update(progress=0, total=1)

    @on(Button.Pressed, "#predict")
    def on_predict(self) -> None:
        """on_predict is called when the Predict button is pressed."""
        # Get the value from the classifier and features
        # do validation and then call display_prediction_results
        classifier = self.query_one("#classifier", Select).value
        text = self.query_one("#input-text", TextArea).text

        self.display_prediction_results(classifier, text)

    # Do not change this function.
    def display_prediction_results(self, classifier: str, text: str) -> None:
        """display_prediction_results calls the ML pipeline and shows o/p in UI."""
        # Given the classifier and features as text,
        # pass it to predict function and get positive and negative probabilities
        # then update positive and negative ; and positive_score, negative_score
        # do not forget to handle errors.
        parsed_text: list[int] = list(map(int, text.strip().split()))
        positive, negative = pipeline_configs.get_prediction_probabilities(parsed_text, classifier)
        self.positive_progress, self.negative_progress = positive, negative
        self.query_one("#positive_score", Label).update(f"Positive Score: {self.positive_progress}")
        self.query_one("#negative_score", Label).update(f"Negative Score: {self.negative_progress}")
        self.query_one("#positive", ProgressBar).update(total=1, progress=self.positive_progress)
        self.query_one("#negative", ProgressBar).update(total=1, progress=self.negative_progress)


if __name__ == "__main__":
    app = TextualClassifier()
    app.run()
