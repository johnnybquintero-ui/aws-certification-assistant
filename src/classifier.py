import pickle
from pathlib import Path


class AWSServiceClassifier:
    def __init__(self, model_path: Path) -> None:
        with model_path.open("rb") as model_file:
            self.model = pickle.load(model_file)

    def classify(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            raise ValueError("Input must not be empty.")

        prediction = self.model.predict([cleaned_text])

        return str(prediction[0])

    def classify_with_confidence(
        self,
        text: str,
    ) -> list[tuple[str, float]]:
        """classify_with_confidence takes a string input and returns a list of tuples containing the predicted AWS service and its associated probability."""
        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            raise ValueError("Input must not be empty.")

        probabilities = self.model.predict_proba([cleaned_text])[0]
        classes = self.model.classes_

        ranked_predictions = sorted(
            zip(classes, probabilities),
            key=lambda prediction: prediction[1],
            reverse=True,
        )

        return [
            (str(service), float(probability))
            for service, probability in ranked_predictions[:3]
        ]
