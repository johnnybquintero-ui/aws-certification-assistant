# AWS Certification Assistant

## Project overview

The **AWS Certification Assistant** is a machine learning project that currently uses a natural language classifier to identify AWS services.

It uses machine-readable service data from [Botocore](https://github.com/boto/botocore) to create a dataset describing AWS services and operations.

The classifier will identify the AWS service most relevant to a user's question. It may later be combined with a chatbot and retrieval-augmented generation (RAG) to support AWS learning, troubleshooting and certification study.

The project is currently under development and will expand as new components are implemented.

## Requirements and technologies

The project uses:

* Python;
* Botocore for AWS service data;
* PyArrow for Parquet files;
* Beautiful Soup for cleaning HTML;
* pytest for automated testing.
* pandas for working with the processed dataset;
* scikit-learn for classifier training and evaluation;
* Matplotlib for confusion matrix visualisation;

The exact dependency versions are recorded in `requirements.txt`.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Running the tests

Run the test suite from the project root:

```bash
python -m pytest
```

## Ingestion, transformation and export

The current data pipeline:

* loads AWS service models from Botocore;
* extracts service and operation information;
* saves the original records as a raw Parquet file;
* cleans and standardises the records;
* saves the cleaned records as a processed Parquet file.

The ingestion code is contained in `src/ingest.py`.

The transformation code in `src/transform.py`:

* removes HTML from descriptions;
* normalises whitespace;
* standardises service identifiers;
* checks that required fields are present;
* removes duplicate operations.

The export code in `src/export.py` saves the records using the Parquet format.

`src/data_pipeline.py` runs the complete data prep pipeline.

The generated files are:

```text
data/raw/operations.parquet
data/processed/operations_clean.parquet
```

## Training the classifier

The classifier uses TF-IDF vectorisation and Logistic Regression to predict an AWS service from a natural-language requirement or operation description.

Training uses two data sources:

Processed Botocore operation descriptions.
data/service_intents.csv, containing 504 curated synthetic intents across 21 AWS services.

The intent examples supplement the formal Botocore language with shorter, user-style descriptions of AWS capabilities. They are balanced across the service classes and avoid including service names in the descriptions.

Generate the processed dataset before training:

```bash
python -m src.data_pipeline
```

Train and evaluate the classifier:

```bash
python -m src.train_model
```

The training process uses a stratified 80/20 train-test split and evaluates accuracy, macro precision, macro recall and macro F1 score. It also logs a classification report and generates a normalised confusion matrix.

The trained pipeline is saved using pickle. The generated model files are:

```text
models/aws_service_classifier.pkl
models/confusion_matrix.png
```

### Natural-language intent data

The Botocore dataset primarily contains formal AWS API documentation, which differs from the conversational questions expected from chatbot users. To reduce this gap, a curated intent dataset was added containing example requirements such as:

```text
I need a managed MySQL database for a web application. → rds
I need somewhere to upload and download millions of files. → s3
```

The intent dataset is split into separate training and test sets. Only its training split is combined with the Botocore training data, preventing test examples from leaking into model training.

Intent augmentation can be enabled with:

```bash
python -m src.train_model --include-intents
```

Both model variants are evaluated against the held-out Botocore and intent test sets. Adding intent data improved accuracy on conversational requests from **46.5% to 73.3%**, making the enhanced classifier more suitable for the natural-language interface.


## Natural Language Interface

The chatbot uses a **Classify Then Reply** architecture. Free-form user input is passed directly to the trained classifier, which predicts the most relevant AWS services. These predictions are then passed to a pretrained language model to produce a friendly response.

```text
User input → AWS service classifier → Language model response
```

This approach keeps service selection controlled by the classifier while using the language model only to explain the result conversationally.

Ensure the trained model exists, then start the chatbot from the project root:

```bash
python -m src.chatbot_interface
```

Enter a description of an AWS requirement when prompted:

```text
I need a managed relational database.
```

Enter `exit` or `quit` to close the chatbot.


## Licensing

Botocore is maintained and published by Amazon Web Services and is licensed under the [Apache License 2.0](https://github.com/boto/botocore/blob/develop/LICENSE.txt).

The Apache License 2.0 permits Botocore material to be used, modified and redistributed. Any Botocore material included or adapted by this project will retain the required licence and copyright notices.

The licence for the original code in this project is documented separately in the repository's `LICENSE` file.

This is an independent educational project and is not affiliated with or endorsed by Amazon Web Services.

## Citation

This project uses service data provided by Botocore:

> Amazon Web Services. *Botocore: the low-level, core functionality of Boto3 and the AWS CLI.*
> https://github.com/boto/botocore
> Licensed under the Apache License, Version 2.0.

The exact Botocore version used to generate the dataset is recorded in `requirements.txt`.
