# AWS Certification Assistant

## Project title and description

The **AWS Certification Assistant** is a machine learning project that combines a natural language classifier with a pre-trained language model.

The project will use machine-readable service data from [Botocore](https://github.com/boto/botocore) to build a dataset describing AWS services, operations, parameters and errors. A simple classifier will be trained to identify the AWS service most relevant to a user's question.

The classifier will later be combined with a chatbot and retrieval-augmented generation (RAG). The aim is to help users understand AWS services, explore troubleshooting scenarios and support their AWS certification studies.

This project is currently in the planning and development stage. Its scope will be expanded as each part is implemented.

## Licensing

Botocore is maintained and published by Amazon Web Services and is licensed under the [Apache License 2.0](https://github.com/boto/botocore/blob/develop/LICENSE.txt).

The Apache License 2.0 permits the Botocore material to be used, modified and redistributed. Any Botocore material included or adapted by this project will retain the required licence and copyright notices, and any modifications will be identified.

The licence for the original code in this project has not yet been selected. It will be documented separately in a `LICENSE` file.

This is an independent educational project and is not affiliated with or endorsed by Amazon Web Services.

## Citations

This project uses service data provided by Botocore:

> Amazon Web Services. *Botocore: the low-level, core functionality of Boto3 and the AWS CLI.*  
> https://github.com/boto/botocore  
> Licensed under the Apache License, Version 2.0.

The exact Botocore version used to generate the dataset will be added here once the ingestion process has been implemented.
