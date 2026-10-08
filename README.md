# Final_Capstone

# Salama: NLP-Based Gender-Based Violence Reporting and Referral System

## Project Overview

Gender-Based Violence (GBV) remains a serious social and public health concern in Kenya. Survivors may experience physical, sexual, emotional, psychological or economic abuse, but accessing appropriate support can be difficult due to stigma, fear, privacy concerns and limited awareness of available services.

**Salama** is a proposed Natural Language Processing (NLP) system designed to analyse written GBV reports, identify different forms of violence, detect potential crisis indicators and recommend appropriate support services.

The project aims to demonstrate how machine learning can support humanitarian organisations and GBV response initiatives through more accessible, privacy-conscious reporting and referral processes.

## Problem Statement

GBV survivors may struggle to describe their experiences using predefined reporting categories. A single report may involve multiple forms of abuse, threats or urgent safety concerns.

Traditional reporting forms may not adequately capture these overlapping experiences or help users understand which services could be relevant to their situation.

This project aims to build an NLP-based system that can interpret written reports, classify the types of violence described and provide appropriate referral guidance while prioritising survivor privacy and choice.

## Project Objectives

The main objectives are to:

1. Develop an NLP model to classify different forms of GBV from written narratives.
2. Identify potential crisis indicators, including threats, serious injuries and immediate safety concerns.
3. Compare traditional machine-learning approaches with advanced transformer-based NLP models.
4. Implement privacy protection through the detection and masking of personally identifiable information.
5. Develop a referral recommendation system for relevant support services.
6. Build an interactive application where users can submit sample reports and receive guidance.

## Proposed GBV Categories

The project will investigate classification into the following categories:

- **Physical violence:** Hitting, assault and other forms of physical harm.
- **Sexual violence:** Sexual assault, coercion and other non-consensual sexual acts.
- **Emotional and psychological abuse:** Intimidation, humiliation, manipulation and verbal abuse.
- **Economic abuse:** Financial control, withholding resources and restricting financial independence.
- **Threats and coercive control:** Threats, intimidation, isolation and controlling behaviour.
- **Stalking and harassment:** Repeated unwanted contact, monitoring and harassment.

The final categories will depend on the labels available in the selected datasets and any additional annotation work.

Because multiple forms of abuse can occur in one report, the project will explore **multi-label classification**.

## Dataset

The project will begin by exploring **CRADLE Bench**, a dataset containing annotated narratives relating to interpersonal violence and crisis situations.

Additional GBV-related datasets may be investigated to improve category coverage and model evaluation.

The dataset preparation stage will include:

- Reviewing available text and labels.
- Checking missing values and duplicate records.
- Examining class distributions.
- Identifying overlapping categories.
- Assessing dataset relevance and limitations.

Only appropriately licensed or authorised data will be used. The application prototype will use fictional or safely de-identified examples rather than collecting real survivor reports.

## Methodology

### Step 1: Data Exploration and Preprocessing

Explore the datasets to understand their structure, labels and text characteristics.

Text preprocessing may include cleaning unnecessary characters, handling missing values and preparing text for tokenisation while preserving meaningful information such as negation and threat-related expressions.

### Step 2: Baseline Machine-Learning Model

Develop a baseline NLP classifier using:

- TF-IDF vectorisation
- Logistic Regression

The baseline will provide a reference for evaluating more advanced models.

### Step 3: Advanced NLP Model

Fine-tune a pretrained transformer model such as **DistilBERT or BERT** for GBV text classification.

The transformer model will be compared against the baseline to determine whether it improves classification performance.

### Step 4: Model Evaluation

Evaluate the classification models using:

- Precision
- Recall
- F1-score
- Confusion matrices
- Per-category performance
- Error analysis

For multi-label classification, appropriate micro- and macro-averaged metrics will be used.

Particular attention will be given to false negatives, where relevant GBV-related information may not be detected.

### Step 5: Crisis Indicator Detection

Develop a separate component to identify potential safety concerns, such as:

- Threats of serious harm
- References to weapons
- Serious injuries
- Statements suggesting immediate danger

This component will support safety-related guidance rather than make definitive assessments of a person's level of risk.

### Step 6: Privacy Protection

Explore NLP-based detection and masking of personally identifiable information, including names, telephone numbers and addresses.

The application will be designed to minimise unnecessary data collection and avoid storing sensitive reports by default.

### Step 7: Referral Recommendation System

Develop a rule-based referral component that suggests relevant types of assistance based on the information identified in a report.

Possible referral categories include:

- Medical and healthcare support
- Counselling and psychosocial support
- Legal assistance
- Safe shelters
- GBV-focused NGOs
- Emergency support services

The project will use simulated referral workflows. Any sharing of report information will require explicit user consent.

### Step 8: Application Development

Build an interactive prototype using **Streamlit**.

The application will allow users to enter fictional sample reports and view:

- Predicted GBV categories
- Potential crisis indicators
- Privacy-masked text
- Suggested support-service categories
- An optional simulated referral request

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Pandas and NumPy | Data preparation and analysis |
| Matplotlib and Seaborn | Data visualisation |
| Scikit-learn | Baseline NLP modelling and evaluation |
| Hugging Face Transformers | Advanced NLP modelling |
| PyTorch | Transformer training |
| spaCy / Regex | Personal information detection |
| Streamlit | Application development |

## Expected Outcome

The expected outcome is a functional NLP prototype capable of:

1. Identifying multiple forms of GBV in written reports.
2. Flagging potential crisis indicators.
3. Masking selected personal information.
4. Recommending relevant support-service categories.
5. Demonstrating a privacy-conscious and survivor-controlled referral process.

The project will also compare baseline and transformer-based models and document their limitations, particularly when applied to sensitive real-world scenarios.

## Possible Future Improvements

Depending on data availability and project progress, future improvements may include:

- Kiswahili language support.
- Evaluation using Kenyan-context narratives.
- Integration with verified local support-service directories.
- Improved detection of coercive control and indirect threats.
- Human-reviewed referral workflows.
- Secure deployment with appropriate safeguarding measures.

## Ethical Considerations

Because GBV reporting involves highly sensitive information, the project will prioritise privacy, consent and survivor autonomy.

The system will not be presented as a replacement for trained GBV professionals or emergency responders.

Predictions may be incorrect, and the absence of a detected crisis indicator must never be interpreted as proof that a survivor is safe.

All demonstrations will use fictional or appropriately de-identified data, and no real emergency notifications will be sent.
