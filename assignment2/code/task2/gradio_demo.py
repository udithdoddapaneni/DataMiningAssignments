# GiG

import gradio as gr

from ds5612_pa2.code import pipeline_configs


CLASSIFIER_OPTIONS = [clf.value for clf in pipeline_configs.ValidClassifierNames]


def predict(classifier: str, item_str: str) -> tuple[dict[str, float], str]:
    """Predict using the specified classifier and input values."""
    class_names = ["Positive", "Negative"]
    probabilities = [0.0, 1.0]
    error_msg = "Success"
    try:
        item = [float(elem) for elem in item_str.split()]
        probabilities = pipeline_configs.get_prediction_probabilities(item, classifier)
    except Exception as e:  # noqa: BLE001
        error_msg = str(e)
    return {class_names[i]: probabilities[i] for i in range(2)}, error_msg


# Create Gradio interface
####################################Only change here
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(label="Classifier", value="DecisionTree", choices=CLASSIFIER_OPTIONS),
        gr.Text(label="Features"),
    ],
    outputs=[gr.Label(label="output 0"), gr.Text(label="output 1", interactive=False)],
    title="ML Classifier",
    description="Choose a classifier and enter features to get prediction probabilities.",
)
####################################Only change here

if __name__ == "__main__":
    # Launch the app
    iface.launch()
