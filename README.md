# Final_Capstone

# Salama: NLP-Based Gender-Based Violence Support Recommendation System

## Project Overview

Gender-Based Violence (GBV) remains a serious social and public health concern in Kenya. Individuals experiencing violence may face challenges accessing appropriate support due to stigma, fear, privacy concerns, and limited awareness of available services.

**Salama** is an NLP-based support recommendation system designed to help individuals identify relevant GBV support resources based on their experiences or preferred type of assistance.

The system combines **multi-label text classification, Kiswahili-to-English translation, and a rule-based recommendation engine** to recommend relevant support services.

Salama provides two ways of accessing support:

1. **Browse Support Services:** Users can explore available support services without describing their experiences.
2. **Share Your Experience:** Users can describe their experiences in English or Kiswahili and receive recommendations based on the types of GBV identified.

The project demonstrates how machine learning and Natural Language Processing can support more accessible, privacy-conscious referral processes in the Kenyan context.

---

## Problem Statement

Gender-Based Violence survivors often face difficulties identifying appropriate support services.

Existing support resources may be distributed across different organisations, making it challenging for individuals to determine where to seek assistance.

Additionally, survivors may experience multiple forms of violence simultaneously, making it difficult to identify relevant services through traditional predefined reporting categories.

This project addresses these challenges by developing an NLP-based recommendation system that:

- Analyses written descriptions of GBV experiences.
- Identifies potentially relevant forms of violence.
- Recommends appropriate categories of support services.
- Provides information about available support organisations.
- Allows users to access support information without disclosing personal experiences.

---

## Project Objectives

The main objectives of Salama are to:

1. Develop an NLP model capable of identifying multiple forms of GBV from written narratives.
2. Implement multi-label classification using a fine-tuned transformer model.
3. Support English and Kiswahili narratives through multilingual processing.
4. Develop a rule-based recommendation engine that maps identified GBV categories to relevant support services.
5. Integrate Nairobi-based GBV support organisations and national support helplines.
6. Build an interactive and mobile-responsive Streamlit application.
7. Promote privacy-conscious access to support information.

---

## GBV Classification Categories

Salama uses a multi-label classification model to identify four categories:

| Category | Description |
|----------|-------------|
| Domestic Violence | Violence or abuse occurring within domestic or intimate relationships |
| Rape | Narratives involving rape |
| Sexual Harassment | Unwanted sexual behaviour or harassment |
| Child Abuse and Endangerment | Narratives involving abuse or endangerment of children |

Because multiple forms of violence can occur within one experience, the model uses **multi-label classification**.

This allows a single narrative to be associated with more than one category.

The absence of a predicted category does not necessarily mean that violence has not occurred.

---

## Dataset

### 1. GBV Classification Dataset

The NLP classification component uses data from **CRADLE Bench**, which contains annotated narratives relating to interpersonal violence and other sensitive experiences.

Dataset preparation involved:

- Exploring the dataset structure.
- Checking missing values and duplicate records.
- Analysing class distributions.
- Preparing the text for tokenisation.
- Encoding target labels for multi-label classification.
- Splitting the dataset for model training and evaluation.

### 2. GBV Support Resources

The recommendation system uses support-resource information compiled from a Nairobi County GBV referral directory.

The resources include organisations offering:

- Psychosocial Support
- Healthcare
- Legal Services
- Shelter and Security
- Economic Empowerment

The application also includes national support helplines.

The referral datasets are stored locally as CSV files:

- `gbv_support_resources.csv`
- `nairobi_general_support.csv`
- `national_gbv_hotlines.csv`

These datasets provide the information used by the recommendation engine.

**Note:** Contact details and service availability may change and should be verified before real-world use.

---

## Methodology

### Step 1: Data Exploration and Preprocessing

The first stage involved exploring the GBV classification dataset to understand its structure and characteristics.

The preprocessing process included:

- Handling missing values.
- Removing duplicate records where appropriate.
- Examining label distributions.
- Preparing text for transformer tokenisation.
- Encoding multiple target labels.

The prepared dataset was used for model development and evaluation.

### Step 2: Baseline Machine Learning

Traditional machine-learning approaches were explored to establish a baseline for GBV text classification.

These included:

- TF-IDF vectorisation
- Logistic Regression

The baseline provided a reference for evaluating the transformer-based approach.

### Step 3: Advanced NLP Classification

A pretrained **DistilBERT** transformer model was fine-tuned for multi-label GBV classification.

The model processes written narratives and produces probabilities for the four GBV categories.

A sigmoid activation function converts model outputs into independent probabilities.

Category-specific classification thresholds are then applied to determine the predicted labels.

This approach allows multiple GBV categories to be identified within a single narrative.

### Step 4: Model Evaluation

The classification model was evaluated using metrics appropriate for multi-label classification.

These included:

- Precision
- Recall
- F1-score
- Per-category evaluation
- Micro- and macro-averaged metrics
- Error analysis

The fine-tuned DistilBERT model achieved approximately **0.784 macro F1** on its English validation dataset.

However, this performance does not establish the model's accuracy on representative Kenyan narratives or Kiswahili inputs.

### Step 5: Kiswahili Language Support

To make Salama more accessible in the Kenyan context, the application supports both English and Kiswahili.

For Kiswahili narratives, the system uses the **NLLB-200 distilled 600M** translation model.

The translation process follows these steps:

1. The user enters a narrative in Kiswahili.
2. The NLLB model translates the narrative into English.
3. The translated text is passed to the DistilBERT classifier.
4. The predicted categories are used by the recommendation engine.

Both translation and classification use locally stored models.

Translation may alter or omit important details, and the Kiswahili pipeline requires further evaluation.

### Step 6: Support Recommendation Engine

Salama implements a **rule-based recommendation system**.

The recommendation engine maps predicted GBV categories to relevant support services.

For example:

| GBV Category | Recommended Support Services |
|--------------|------------------------------|
| Domestic Violence | Psychosocial Support, Shelter & Security, Healthcare, Legal Services |
| Rape | Healthcare, Psychosocial Support, Legal Services, Shelter & Security |
| Sexual Harassment | Psychosocial Support, Legal Services, Healthcare |
| Child Abuse and Endangerment | Shelter & Security, Healthcare, Psychosocial Support, Legal Services |

The system retrieves relevant organisations from the support-resource directory.

Users can also browse organisations directly by selecting a support-service category.

This means that users are not required to describe their experiences to access support information.

The recommendation engine does not perform real-time verification of organisational availability or assess the urgency of an individual's situation.

### Step 7: Streamlit Application Development

An interactive web application was developed using **Streamlit**.

The application allows users to:

- Select English or Kiswahili.
- Choose their location.
- Browse support services without providing a narrative.
- Describe an experience in their preferred language.
- Receive recommendations for relevant support organisations.
- View organisational contact information.
- Access national GBV support helplines.
- Clear their narrative input.

---

## System Architecture

The Salama system follows the workflow below:

### Narrative-Based Recommendation

```text
User Enters Narrative
         |
         v
Language Selection
         |
         v
Kiswahili-to-English Translation
(Only for Kiswahili Input)
         |
         v
DistilBERT Multi-Label Classifier
         |
         v
Predicted GBV Categories
         |
         v
Rule-Based Recommendation Engine
         |
         v
Relevant Support Organisations
         |
         v
Support Contacts Displayed
```

### Direct Support Browsing

```text
User Selects Support Service
         |
         v
Rule-Based Recommendation Engine
         |
         v
Support Resource Directory
         |
         v
Relevant Organisations Displayed
```

---

## Tools and Technologies

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas and NumPy | Data processing and analysis |
| Matplotlib and Seaborn | Data visualisation |
| Scikit-learn | Baseline modelling and evaluation |
| Hugging Face Transformers | Transformer-based NLP models |
| PyTorch | Model training and inference |
| DistilBERT | Multi-label GBV classification |
| NLLB-200 | Kiswahili-to-English translation |
| Streamlit | Interactive web application |
| CSV Files | Support-resource storage |

---

## Application Features

### 1. Bilingual Interface

Salama supports English and Kiswahili, allowing users to interact with the application in their preferred language.

### 2. Multi-Label GBV Classification

The system can identify multiple potentially relevant GBV categories from a single narrative.

### 3. Personalised Support Recommendations

Recommendations are generated based on the GBV categories identified in the user's narrative.

### 4. Direct Support Browsing

Users can access support information without describing their experiences.

### 5. Location-Based Resource Availability

The application currently provides Nairobi-based organisational referrals.

Users in other counties can access national support helplines.

### 6. National Support Contacts

National GBV support helplines remain accessible regardless of classification results.

### 7. Mobile-Responsive Interface

The Streamlit application adapts to smaller screens, allowing users to navigate it on mobile devices.

### 8. Offline Model Processing

The application can run locally using downloaded models without requiring external AI inference APIs.

---

## Ethical Considerations

Because GBV experiences involve sensitive personal information, Salama is designed with privacy, safety, and user choice in mind.

The application:

- Allows users to browse support services without sharing a narrative.
- Encourages users not to include names or identifying information.
- Provides an option to clear narrative input.
- Uses locally stored models for classification and translation.
- Displays national support contacts independently of model predictions.

However, Salama is an informational recommendation prototype and does not replace trained GBV professionals, medical services, or emergency responders.

The current prototype does not include automated personal-information masking, validated crisis detection, or emergency intervention.

Clearing a narrative from the interface does not guarantee deletion from all browser, server, or network environments.


---

## Current Limitations

1. The classification model has not been validated on representative Kenyan GBV narratives.
2. Kiswahili classification depends on machine translation quality.
3. Organisation-specific referrals are currently limited to Nairobi.
4. The recommendation engine uses predefined rules rather than a learned ranking model.
5. Support-resource contact details and availability are not verified in real time.
6. The system does not provide professional risk assessment or emergency intervention.

---

## Future Improvements

Potential improvements include:

- Expanding support-resource coverage to additional Kenyan counties.
- Evaluating the system using appropriately governed Kenyan-context narratives.
- Improving multilingual classification and translation.
- Introducing more advanced recommendation techniques.
- Integrating regularly verified support-resource directories.
- Developing human-reviewed referral workflows.
- Strengthening privacy and security protections.
- Deploying the application securely for wider accessibility.

---

## Expected Outcome

Salama demonstrates how **Natural Language Processing, multi-label classification, multilingual translation, and rule-based recommendations** can be integrated into a support-resource recommendation system.

The project aims to make GBV support information more accessible by helping individuals identify potentially relevant services while maintaining the option to seek information without sharing personal experiences.
