# PCIC at SMM4H 2024: Enhancing Reddit Post Classification on Social Anxiety Using Transformer Models and Advanced Loss Functions

## This repo hosts the code for the ACL Paper publication: https://aclanthology.org/2024.smm4h-1.14/

## Overview
This repository hosts the materials and code for Task 3 of the 9th Social Media Mining for Health Research and Applications Workshop (SMM4H) at ACL. The project focuses on using Large Language Models (LLMs) to analyze the effects of outdoor activities on social anxiety, based on Reddit posts from r/socialanxiety.

## Project Description
The project's primary goal is to classify Reddit posts into categories reflecting the impact of outdoor spaces on social anxiety levels. This classification helps in understanding how different environments can affect individuals suffering from social anxiety. The project initially focuses on a one-step classification model to handle this multi-class classification task efficiently.

## Classes Description
The posts are classified into four categories:
- **Class 0 (Unrelated):** Posts that do not relate to the impact of outdoor activities on social anxiety.
- **Class 1 (Positive Effect):** Posts indicating that outdoor activities have a positive effect on the poster's social anxiety.
- **Class 2 (Neutral Effect):** Posts discussing outdoor activities without a clear positive or negative impact on social anxiety.
- **Class 3 (Negative Effect):** Posts suggesting that outdoor activities have worsened the poster's social anxiety.

## Dataset
The dataset comprises annotated posts from the subreddit r/socialanxiety, specifically curated for this task. It includes:
- **Training set:** 1800 annotated posts.
- **Validation set:** 600 annotated posts.

## Models and Training
This project utilizes several transformer-based models for the classification task:
- BERT
- RoBERTa
- XLNet
- DistilBERT
- XLM-Roberta

These models were fine-tuned on the provided dataset, focusing on achieving the best performance in distinguishing between the four classes through various techniques like class weighting and data augmentation.

## Repository Structure
- `data/`: Contains training and validation datasets.
- `models/`: Script for the transformer architectures employed.
- `utils/`: Utility scripts for data augmentation, preprocessing, and additional functionalities.
- `main.ipynb`: Jupyter notebook with the implementation of the one-step classification process.
- `README.md`: Documentation providing an overview and guide for the repository.

## Usage
To replicate the results or to use this project as a base for further research:
1. Clone this repository.
2. Install required dependencies.
3. Run the `main.ipynb` notebook for a detailed step-by-step explanation of the data preparation, model training, and evaluation process.

## Additional Information
This code was developed as part of the participation in the ACL workshop SMM4H: The 9th Social Media Mining for Health Research and Applications Workshop and Shared Tasks. The specific focus of this repository is Task 3, which centers on the generalizability of large language models for social media NLP, particularly in analyzing user-generated content from social media to study health-related outcomes in the context of social anxiety.

### Note
While the repository primarily focuses on the one-step classification approach, additional exploratory work on a two-step classification process is also included. This secondary focus is aimed at experimenting with hierarchical classification strategies to potentially enhance the model's performance and interpretability.

## Contribution
Contributions to this project are welcome. You can suggest changes by forking this repository, making changes, and submitting a pull request.
