# AWS Certification Assistant

## Project overview

The **AWS Certification Assistant** is a machine learning project combining a natural language classifier with a pre-trained language model.

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

## Running the pipeline

Run the pipeline from the project root:

```bash
python main.py
```

## Running the tests

Run the test suite from the project root:

```bash
python -m pytest
```

The tests cover ingestion, transformation and Parquet export.

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

`main.py` runs the complete pipeline.

The generated files are:

```text
data/raw/operations.parquet
data/processed/operations_clean.parquet
```
## Training the classifier

The classifier uses TF-IDF vectorisation and Logistic Regression to predict an AWS service from an operation description.

Generate the processed dataset before training:

```bash
python main.py
```

Train and evaluate the classifier:

```bash
python -m src.train_model
```

The training process uses a stratified 80/20 train-test split and evaluates accuracy, macro precision, macro recall and macro F1 score. It also logs a classification report and generates a normalised confusion matrix.

The generated model files are:

```text
models/aws_service_classifier.pkl
models/confusion_matrix.png
```

## Running the tests

Run the test suite from the project root:

```bash
python -m pytest
```

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
