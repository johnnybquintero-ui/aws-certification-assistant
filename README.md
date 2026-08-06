# AWS Certification Assistant

## Project Overview

This project is an AWS certification assistant that combines three separate machine learning components:

- A text classifier that recommends an AWS service from a user's natural-language requirements.
- A retrieval-augmented generation (RAG) system that retrieves relevant information from a local AWS knowledge base.
- An instruction-tuned language model that uses the retrieved information to produce a short, natural-language response.

The assistant supports both AWS service recommendation requests and general questions about the AWS Certified Cloud Practitioner exam.

It uses machine-readable service data from [Botocore](https://github.com/boto/botocore) to create a dataset describing AWS services and operations.

## Requirements and technologies

The project uses:

* Python;
* Botocore for AWS service data;
* PyArrow for Parquet files;
* Beautiful Soup for cleaning HTML;
* pytest for automated testing;
* pandas for working with the processed dataset;
* scikit-learn for classifier training and evaluation;
* Matplotlib for confusion matrix visualisation;
* Sentence Transformers for text embeddings;
* Hugging Face Transformers and PyTorch for response generation;
* Accelerate for automatic device allocation.

The exact dependency versions are recorded in `requirements.txt` and `requirements_dev.txt`.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

Install the development and testing dependencies with:

```bash
python -m pip install -r requirements_dev.txt
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

- Processed Botocore operation descriptions.
- `data/service_intents.csv`, containing 504 curated synthetic intents across 21 AWS services.

The intent examples supplement the formal Botocore language with shorter, user-style descriptions of AWS capabilities. They are balanced across the service classes and avoid including service names in the descriptions.

Generate the processed dataset before training:

```bash
python -m src.data_pipeline
```

Train and evaluate the classifier:

```bash
python -m src.train_model --include-intents
```
Omit `--include-intents` to train the Botocore-only baseline for comparison.

The training process uses a stratified 80/20 train-test split and evaluates accuracy, macro precision, macro recall and macro F1 score. It also logs a classification report and generates a normalised confusion matrix.

The trained pipeline is saved using pickle. The generated model files are:

```text
models/aws_service_classifier.pkl
models/confusion_matrix_intents_post_intents.png #confusion matrix heatmap results on intents data
models/confusion_matrix_botocore_post_intents.png #confusion matrix heatmap results on botocore data
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

The chatbot supports two types of request:

- **Service recommendations:** The classifier selects an AWS service, then RAG and the language model explain the result.
- **Exam questions:** The classifier is bypassed and RAG provides context for a direct answer.

The response generator uses `Qwen/Qwen3-4B-Instruct-2507`.

### Run the Chatbot

Ensure the trained classifier exists, then run:

```bash
python -m src.chatbot_interface
```

Example service request:

```text
I need a managed MySQL database for a web application.
```

Example exam question:

```text
What security topics are covered in the Cloud Practitioner exam?
```

Enter `exit` or `quit` to close the chatbot.

> Responses should be verified against official AWS documentation.

## Retrieval-Augmented Generation

The chatbot uses retrieval-augmented generation (RAG) to ground its answers in locally stored AWS reference material.

The RAG pipeline:

1. Loads AWS reference documents from the project's RAG data directory.
2. Splits the documents into smaller text chunks.
3. Converts each chunk into an embedding using a Sentence Transformers model.
4. Converts the user's question into an embedding.
5. Compares the question embedding with the stored document embeddings using cosine similarity.
6. Retrieves the most relevant chunks.
7. Supplies those chunks to the chatbot language model as supporting context.

This separates retrieval from response generation. The retriever finds relevant information, while the language model turns that information into a concise answer.

Document and query embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.

### RAG Data

The knowledge base contains cleaned text derived from authoritative AWS certification and service documentation.

The documents are organised into focused sections so that retrieved chunks contain closely related information rather than large amounts of unrelated exam content.

When adding new RAG material:

- Prefer official AWS documentation and exam guides.
- Record each original source in `data/knowledge_base/clf_c02/clf_c02_source_manifest.csv`.
- Remove repeated navigation, metadata and unrelated page content.
- Use descriptive headings to keep chunks topic-specific.
- Do not rely on the language model's existing knowledge as factual evidence.

The local knowledge-base documents are stored in:

`data/knowledge_base/`

### Logging

Chatbot interactions are recorded in `logs/chatbot_history.log` for later review and are not displayed in the terminal.

## Current Limitations

- The classifier can only recommend services represented in its training data.
- Natural-language predictions may have relatively low confidence when the training dataset is small.
- RAG answers are limited by the coverage of the local knowledge base.
- Retrieved context may not contain enough detail to answer every service-specific question.
- Language-model output can still contain errors, so responses should be checked against official AWS documentation.
- The Qwen model requires more memory and has a slower startup time than TinyLlama.
- Performance depends on the user's available CPU, GPU and system memory.


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

The sources used to build the RAG knowledge base are recorded in `data/knowledge_base/clf_c02/clf_c02_source_manifest.csv`. The knowledge-base documents were cleaned and reformatted from official AWS certification and service documentation.