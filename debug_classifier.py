from src.classifier import AWSServiceClassifier
from src.chatbot_interface import MODEL_PATH

classifier = AWSServiceClassifier(MODEL_PATH)

print("Model path:", MODEL_PATH.resolve())
print("Classes:", classifier.model.classes_)

print(
    classifier.classify_with_confidence(
        "I need protection against bots and automated attacks"
    )
)
