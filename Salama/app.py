import streamlit as st

import sys
from pathlib import Path

# Allow Streamlit to import our recommendation engine
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(MODELS_DIR))

import recommendation_engine

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths to our saved model and tokenizer
MODEL_DIR = MODELS_DIR / "multilabel_model"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"


@st.cache_resource
def load_gbv_model():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


tokenizer, gbv_model = load_gbv_model()

# GBV categories
gbv_labels = [
    "domesticviolence",
    "rape",
    "sexualharassment",
    "childabuse_endangerment"
]

# Tuned classification thresholds
thresholds = {
    "domesticviolence": 0.50,
    "rape": 0.25,
    "sexualharassment": 0.40,
    "childabuse_endangerment": 0.90
}


def predict_gbv(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = gbv_model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0].tolist()

    predicted_categories = [
        label
        for label, probability in zip(gbv_labels, probabilities)
        if probability >= thresholds[label]
    ]

    return predicted_categories

UI_TEXT = {
    "English": {
        "welcome_heading": "Your safety matters.",
        "welcome_description": "Salama provides a space to describe your experience and explore support resources that may be relevant to you.",
        "welcome_note": "Share only what you feel comfortable sharing. You can also explore support services without describing your experience.",
        "language_label": "Which language would you like to use?",
        "location_label": "Where are you looking for support?",
        "nairobi": "Nairobi",
        "other_counties": "Other counties in Kenya",
        "services_heading": "Explore support services",
        "services_description": "Find support without describing your experience.",
        "service_label": "What kind of support are you looking for?",
        "select_service": "Select a service",
        "psychosocial": "Psychosocial Support",
        "healthcare": "Healthcare",
        "legal": "Legal Services",
        "shelter": "Shelter & Security",
        "economic": "Economic Empowerment",
        "browse_button": "Browse Support Services",
        "narrative_heading": "Share your experience",
        "narrative_description": "Describe what happened in your own words. You do not need to include names or personal details.",
        "narrative_label": "Your experience",
        "placeholder": "Describe your experience here...",
        "find_button": "Find Support",
        "select_service_warning": "Please select a support service.",
        "empty_narrative_warning": "Please describe your experience before continuing.",
        "searching": "Finding relevant support resources...",
        "results_heading": "Support resources",
        "available_resources": "Available support resources",
        "contact": "Contact",
        "services": "Services",
        "no_resources": "No matching local resources were found. You can still use the national helplines below.",
        "outside_nairobi": "County-specific organisation referrals are currently available for Nairobi only. National support contacts are shown instead.",
        "national_heading": "National Support Helplines",
        "national_description": "You can access these contacts without sharing your experience.",
        "helpline_note": "Contact details and availability should be verified.",
        "find_support_heading": "Find the support you need.",
        "safety_heading": "Before you begin",
        "safety_description": "Salama provides informational referrals, not emergency intervention, medical diagnosis, or professional counselling. You do not need to share names or identifying details."
    },

    "Kiswahili": {
        "welcome_heading": "Usalama wako ni muhimu.",
        "welcome_description": "Salama inakupa nafasi ya kueleza ulichopitia na kutafuta huduma za msaada zinazoweza kukufaa.",
        "welcome_note": "Eleza tu yale unayojisikia huru kushiriki. Unaweza pia kutafuta huduma za msaada bila kueleza ulichopitia.",
        "language_label": "Ungependa kutumia lugha gani?",
        "location_label": "Unatafuta msaada katika eneo gani?",
        "nairobi": "Nairobi",
        "other_counties": "Kaunti nyingine nchini Kenya",
        "services_heading": "Tafuta huduma za msaada",
        "services_description": "Unaweza kutafuta msaada bila kueleza ulichopitia.",
        "service_label": "Unahitaji msaada wa aina gani?",
        "select_service": "Chagua huduma",
        "psychosocial": "Msaada wa kisaikolojia na kijamii",
        "healthcare": "Huduma za afya",
        "legal": "Msaada wa kisheria",
        "shelter": "Makazi salama na ulinzi",
        "economic": "Uwezeshaji wa kiuchumi",
        "browse_button": "Tafuta huduma",
        "narrative_heading": "Eleza ulichopitia",
        "narrative_description": "Eleza kilichotokea kwa maneno yako mwenyewe. Si lazima utaje majina au taarifa zako binafsi.",
        "narrative_label": "Ulichopitia",
        "placeholder": "Eleza ulichopitia hapa...",
        "find_button": "Tafuta Msaada",
        "select_service_warning": "Tafadhali chagua aina ya huduma.",
        "empty_narrative_warning": "Tafadhali eleza ulichopitia kabla ya kuendelea.",
        "searching": "Tunatafuta huduma za msaada zinazofaa...",
        "results_heading": "Huduma za msaada",
        "available_resources": "Huduma za msaada zinazopatikana",
        "contact": "Mawasiliano",
        "services": "Huduma",
        "no_resources": "Hakuna huduma za eneo hili zilizopatikana. Bado unaweza kutumia nambari za msaada za kitaifa zilizo hapa chini.",
        "outside_nairobi": "Kwa sasa, taarifa za mashirika kulingana na kaunti zinapatikana Nairobi pekee. Mawasiliano ya msaada ya kitaifa yanaonyeshwa badala yake.",
        "national_heading": "Nambari za Msaada za Kitaifa",
        "national_description": "Unaweza kupata mawasiliano haya bila kueleza ulichopitia.",
        "helpline_note": "Tafadhali hakikisha kuwa mawasiliano haya na huduma zake bado zinapatikana.",
        "find_support_heading": "Pata msaada unaohitaji.",
        "safety_heading": "Kabla ya kuanza",
        "safety_description": "Salama hutoa taarifa kuhusu huduma za msaada, lakini si huduma ya dharura, utambuzi wa kitabibu, au ushauri wa kitaalamu. Si lazima utoe majina au taarifa zinazoweza kukutambulisha."
    }
}


# Page configuration
st.set_page_config(
    page_title="Salama | Find Support",
    page_icon="🪻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""<
<style>
    .stApp {
        background-color: #F8F6F9;
        color: #48404E;
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #48404E;
    }

    .salama-subtitle {
        color: #665A6C;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .welcome-card {
        background-color: #E8E1EC;
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .welcome-card h2 {
        margin-top: 0;
        color: #48404E;
    }

    .welcome-card p {
        color: #48404E;
        line-height: 1.7;
    }

    .privacy-note {
        background-color: #F0ECF2;
        border-left: 4px solid #82718C;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        color: #48404E;
    }
</style>
""", unsafe_allow_html=True)


# Application header
st.title("Salama")
st.markdown(
    '<p class="salama-subtitle">'
    'GBV Support and Resource Recommendation System'
    '</p>',
    unsafe_allow_html=True
)

# Language selection
def clear_on_language_change():
    st.session_state["salama_narrative"] = ""


language = st.selectbox(
    "Language / Lugha",
    options=["English", "Kiswahili"],
    index=0,
    on_change=clear_on_language_change
)

# Select the appropriate interface translations
t = UI_TEXT[language]

# Welcome section
st.markdown(f"""
<div class="welcome-card">
    <h2>{t['welcome_heading']}</h2>
    <p>{t['welcome_description']}</p>
    <p>{t['welcome_note']}</p>
</div>
""", unsafe_allow_html=True)


# Privacy and safety information
st.markdown(f"""
<div class="privacy-note">
    <strong>{t['safety_heading']}</strong><br>
    {t['safety_description']}
</div>
""", unsafe_allow_html=True)

# Adding Language and Location selection
st.markdown(f"### {t['find_support_heading']}")


# Location selection
location_options = {
    t["nairobi"]: "Nairobi",
    t["other_counties"]: "Other"
}

selected_location = st.selectbox(
    t["location_label"],
    options=list(location_options.keys()),
    index=0
)

# Keep the original values for the recommendation engine
location = location_options[selected_location]

st.caption(
    "Salama currently provides Nairobi-based organisation referrals "
    "and national support helplines for other counties."
)

st.markdown(f"### {t['services_heading']}")

st.write(t["services_description"])

# Display translated service names while preserving English values
service_options = {
    t["select_service"]: None,
    t["psychosocial"]: "Psychosocial Support",
    t["healthcare"]: "Healthcare",
    t["legal"]: "Legal Services",
    t["shelter"]: "Shelter & Security",
    t["economic"]: "Economic Empowerment"
}

selected_service = st.selectbox(
    t["service_label"],
    options=list(service_options.keys())
)

# Original English value used by the recommendation engine
support_service = service_options[selected_service]

browse_support = st.button(
    t["browse_button"],
    use_container_width=True
)

if browse_support:
    if support_service is None:
        st.warning(t["Please select a support service."])

    elif location != "Nairobi":
        st.info(
            "County-specific organisation referrals are currently "
            "available for Nairobi only. National support contacts "
            "will be displayed instead."
        )

        _, _, national_hotlines = (
            recommendation_engine.load_support_resources()
        )

        st.dataframe(
            national_hotlines[["organisation", "contact"]],
            hide_index=True,
            use_container_width=True
        )

    else:
        resources = recommendation_engine.recommend_by_service(
            service_type=support_service,
            top_n=5
        )

        st.markdown("#### Available support resources")

        for _, resource in resources.iterrows():
            with st.container(border=True):
                st.markdown(f"**{resource['organisation']}**")
                st.write(f"Contact: {resource['contact']}")

# Adding the narrative input
st.markdown(f"### {t['narrative_heading']}")

st.write(t["narrative_description"])


# Clear the narrative before Streamlit creates the text area
if "salama_narrative" not in st.session_state:
    st.session_state.salama_narrative = ""

def clear_narrative():
    st.session_state.salama_narrative = ""
    st.session_state.clear_results = True

# Narrative input
narrative = st.text_area(
    t["narrative_label"],
    placeholder=t["placeholder"],
    height=180,
    max_chars=3000,
    key="salama_narrative"
)

# Submit button
find_support = st.button(
    t["find_button"],
    type="primary",
    use_container_width=True
)

clear_label = (
    "Clear My Input"
    if language == "English"
    else "Futa Nilichoandika"
)

st.button(
    clear_label,
    on_click=clear_narrative,
    use_container_width=True
)

if find_support:
    if not narrative.strip():
        st.warning(t["empty_narrative_warning"])

    else:
        # Convert interface selections into model parameters
        selected_language = "sw" if language == "Kiswahili" else "en"
        selected_county = location

        with st.spinner(t["searching"]):
            recommendations = recommendation_engine.salama_recommend(
                text=narrative,
                county=selected_county,
                language=selected_language,
                top_n=5,
                predict_fn=predict_gbv
            )

        st.markdown(f"### {t['results_heading']}")

        if recommendations is not None and not recommendations.empty:
            for _, resource in recommendations.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{resource['organisation']}**")
                    st.write(f"{t['contact']}: {resource['contact']}")

                    if "matched_services" in resource.index:
                        st.caption(
                            "Services: "
                            + ", ".join(resource["matched_services"])
                        )
        else:
            st.info(
                "No matching local resources were found. "
                "You can still seek support through national helplines."
            )

# National support helplines
st.divider()

st.markdown(f"### {t['national_heading']}")

st.write(t["national_description"])

_, _, national_hotlines = (
    recommendation_engine.load_support_resources()
)

for _, hotline in national_hotlines.iterrows():
    with st.container(border=True):
        st.markdown(f"**{hotline['organisation']}**")
        st.write(f"{t['contact']}: {hotline['contact']}")

st.caption(
    "Helpline information is provided for reference. "
)

st.markdown("""
<style>
.salama-ticker {
    overflow: hidden;
    background: #E8E1EC;
    border-top: 1px solid #C6B8CE;
    border-radius: 12px;
    padding: 14px 0;
    margin-top: 35px;
}

.salama-ticker-text {
    display: inline-block;
    white-space: nowrap;
    color: #48404E;
    font-size: 14px;
    font-weight: 500;
    animation: salama-scroll 24s linear infinite;
}

@keyframes salama-scroll {
    from { transform: translateX(100%); }
    to { transform: translateX(-100%); }
}

.salama-ticker:hover .salama-ticker-text {
    animation-play-state: paused;
}

@media (prefers-reduced-motion: reduce) {
    .salama-ticker-text {
        animation: none;
        white-space: normal;
        text-align: center;
    }
}

/* Subtle Salama wallpaper */
/* Soft warm ivory background */
.stApp {
    background-color: #F7F5F2;
    background-image: none;
}

/* Mobile-friendly layout */
@media screen and (max-width: 600px) {

    .block-container {
        max-width: 100%;
        padding: 1.5rem 1rem 3rem;
    }

    h1 {
        font-size: 2rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    .welcome-card {
        padding: 1.25rem;
        border-radius: 14px;
    }

    .welcome-card p,
    .privacy-note {
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Prevent long contact details from overflowing */
    .stMarkdown,
    .stMarkdown p {
        overflow-wrap: anywhere;
    }

    /* Comfortable input size on mobile */
    .stTextArea textarea,
    .stSelectbox input {
        font-size: 16px !important;
    }

    /* Touch-friendly buttons */
    .stButton button {
        min-height: 48px;
        width: 100%;
        border-radius: 12px;
    }

    /* Responsive footer */
    .salama-ticker {
        margin-top: 1.5rem;
        border-radius: 10px;
    }
}

</style>

<div class="salama-ticker">
    <div class="salama-ticker-text">
        Your safety matters &nbsp; ✦ &nbsp;
        Support is available &nbsp; ✦ &nbsp;
        You deserve dignity and respect &nbsp; ✦ &nbsp;
        Salama — A safe space to find support
    </div>
</div>
""", unsafe_allow_html=True)