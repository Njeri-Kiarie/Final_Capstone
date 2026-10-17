import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Model directory
MODEL_DIR = Path("Salama/models")

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR / "tokenizer"
)

# Load binary GBV classification model
binary_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR / "binary_model"
).to(device)

binary_model.eval()

# Load multi-label GBV classification model
multilabel_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR / "multilabel_model"
).to(device)

multilabel_model.eval()

print("Tokenizer loaded successfully!")
print("Binary GBV model loaded successfully!")
print("Multi-label GBV model loaded successfully!")

# Testing the binary GBV classifier
# Sample narrative
sample_text = (
    "My partner constantly threatens me and sometimes "
    "hits me. I am afraid to speak up and need help."
)

# Tokenize
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

# Make prediction
with torch.no_grad():

    outputs = binary_model(**inputs)

    # Sigmoid for single-output binary classification
    gbv_probability = torch.sigmoid(
        outputs.logits
    ).item()

    predicted_class = int(gbv_probability >= 0.5)

print("\nBinary GBV Classification")
print("Narrative:", sample_text)
print("GBV Probability:", round(gbv_probability, 4))
print("Predicted Class:", predicted_class)

if predicted_class == 1:
    print("Prediction: GBV-related")
else:
    print("Prediction: Other crisis-related")

# Testing the multi-label GBV classifier
# GBV categories in their training order
gbv_categories = [
    "domesticviolence",
    "rape",
    "sexualharassment",
    "childabuse_endangerment"
]

# Thresholds selected during development evaluation
gbv_thresholds = {
    "domesticviolence": 0.50,
    "rape": 0.25,
    "sexualharassment": 0.40,
    "childabuse_endangerment": 0.90
}

# Use the same tokenized narrative from the binary test
with torch.no_grad():

    outputs = multilabel_model(**inputs)

    probabilities = torch.sigmoid(
        outputs.logits
    ).squeeze().cpu().numpy()

print("\nMulti-label GBV Classification")

predicted_categories = []

for category, probability in zip(gbv_categories, probabilities):

    threshold = gbv_thresholds[category]

    print(
        f"{category}: {probability:.4f} "
        f"(threshold: {threshold})"
    )

    if probability >= threshold:
        predicted_categories.append(category)

print("\nPredicted GBV Categories:", predicted_categories)