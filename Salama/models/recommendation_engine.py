import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Kiswahili-to-English translation model
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

translation_tokenizer = None
translation_model = None


def translate_swahili_to_english(text):
    global translation_tokenizer, translation_model

    # Load NLLB only when translation is first requested
    if translation_model is None:
        translation_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL_NAME,
            src_lang="swh_Latn",
            local_files_only=True
        )

        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME,
            local_files_only=True
        )

        translation_model.eval()

    # Split the narrative into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    translations = []

    for sentence in sentences:
        if not sentence:
            continue

        inputs = translation_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            translated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(
                    "eng_Latn"
                ),
                max_length=128
            )

        english_sentence = translation_tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        translations.append(english_sentence)

    return " ".join(translations)

# Salama - GBV Support Recommendation Engine

service_mapping = {
    "domesticviolence": [
        "Psychosocial Support",
        "Shelter & Security",
        "Healthcare",
        "Legal Services"
    ],

    "rape": [
        "Healthcare",
        "Psychosocial Support",
        "Legal Services",
        "Shelter & Security"
    ],

    "sexualharassment": [
        "Psychosocial Support",
        "Legal Services",
        "Healthcare"
    ],

    "childabuse_endangerment": [
        "Shelter & Security",
        "Healthcare",
        "Psychosocial Support",
        "Legal Services"
    ]
}

# Loading the support directories
import pandas as pd
import json
from pathlib import Path

# Locate the project's data folder
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_support_resources():

    # Load all Nairobi GBV support organisations
    resources_df = pd.read_csv(
        DATA_DIR / "gbv_support_resources.csv"
    )

    # Convert stored service lists back to Python lists
    resources_df["services"] = resources_df["services"].apply(
        json.loads
    )

    # Load the 20 general psychosocial support organisations
    general_support_df = pd.read_csv(
        DATA_DIR / "nairobi_general_support.csv"
    )

    general_support_df["services"] = general_support_df["services"].apply(
        json.loads
    )

    # Load national GBV hotlines
    national_hotlines_df = pd.read_csv(
        DATA_DIR / "national_gbv_hotlines.csv"
    )

    return resources_df, general_support_df, national_hotlines_df

# Adding category-based recommendations
def recommend_multiple_categories(gbv_categories, top_n=5):

    # Load Nairobi support organisations
    resources_df, _, _ = load_support_resources()

    # Combine relevant services in priority order
    required_services = []

    for category in gbv_categories:
        for service in service_mapping.get(category, []):
            if service not in required_services:
                required_services.append(service)

    # Return an empty result if no supported category was provided
    if not required_services:
        return pd.DataFrame(
            columns=[
                "organisation",
                "contact",
                "matched_services",
                "service_count"
            ]
        )

    # Identify matching services for each organisation
    recommendations = resources_df.copy()

    recommendations["matched_services"] = recommendations["services"].apply(
        lambda services: [
            service for service in required_services
            if service in services
        ]
    )

    # Keep organisations offering at least one relevant service
    recommendations = recommendations[
        recommendations["matched_services"].apply(len) > 0
    ].copy()

    # Prioritise services according to the mapping
    recommendations["priority"] = recommendations["matched_services"].apply(
        lambda services: min(
            required_services.index(service)
            for service in services
        )
    )

    # Count matching services
    recommendations["service_count"] = recommendations[
        "matched_services"
    ].apply(len)

    # Rank recommendations
    recommendations = recommendations.sort_values(
        by=["priority", "service_count"],
        ascending=[True, False]
    )

    return recommendations[
        [
            "organisation",
            "contact",
            "matched_services",
            "service_count"
        ]
    ].head(top_n).reset_index(drop=True)

# Adding the service-specific recommendations
def recommend_by_service(service_type, top_n=5):

    # Load Nairobi support organisations
    resources_df, _, _ = load_support_resources()

    # Find organisations offering the requested service
    recommendations = resources_df[
        resources_df["services"].apply(
            lambda services: service_type in services
        )
    ].copy()

    # Exclude organisations without listed contacts
    recommendations = recommendations[
        recommendations["contact"].notna()
    ].copy()

    # Count services offered by each organisation
    recommendations["service_count"] = recommendations[
        "services"
    ].apply(len)

    # Sort by service coverage
    recommendations = recommendations.sort_values(
        by=["service_count", "organisation"],
        ascending=[False, True]
    )

    return recommendations[
        ["organisation", "contact", "services", "service_count"]
    ].head(top_n).reset_index(drop=True)

# Adding general support recommendations
def recommend_general_support(county="Nairobi"):

    # Load general support organisations and national hotlines
    _, general_support_df, national_hotlines_df = load_support_resources()

    # Outside Nairobi, return national support contacts
    if county is None or county.strip().lower() != "nairobi":
        return national_hotlines_df[
            ["organisation", "contact"]
        ].reset_index(drop=True)

    # Return all 20 Nairobi psychosocial support organisations
    recommendations = general_support_df.copy()

    # Sort alphabetically
    recommendations = recommendations.sort_values(
        by="organisation"
    )

    return recommendations[
        ["organisation", "contact", "county"]
    ].reset_index(drop=True)

def salama_recommend(
    text,
    county=None,
    preferred_service=None,
    support_type=None,
    top_n=5,
    predict_fn=None,
    language="en"
):

    # Process the narrative according to the selected language
    if language == "sw":
        processed_text = translate_swahili_to_english(text)

    elif language == "en":
        processed_text = text

    else:
        raise ValueError("Supported languages are 'en' and 'sw'.")
    
    # The prediction function will be supplied by our ML model
    if predict_fn is None:
        raise ValueError("A GBV prediction function is required.")

    # Predict GBV categories from the narrative
    predicted_categories = predict_fn(processed_text)

    print("Predicted GBV categories:", predicted_categories)
    print("County:", county if county else "Not provided")

    # Check location first
    if county is None or county.strip().lower() != "nairobi":
        print("\nShowing national support contacts.")
        return recommend_general_support(county=county)

    # General support requested directly
    if support_type == "general":
        print("\nShowing general psychosocial support options.")
        return recommend_general_support(county="Nairobi")

    # Specific service requested
    if preferred_service is not None:
        print("Preferred service:", preferred_service)

        return recommend_by_service(
            service_type=preferred_service,
            top_n=top_n
        )

    # No GBV category predicted
    if not predicted_categories:
        print("\nShowing general psychosocial support options.")
        return recommend_general_support(county="Nairobi")

    # Category-based recommendations
    return recommend_multiple_categories(
        gbv_categories=predicted_categories,
        top_n=top_n
    )