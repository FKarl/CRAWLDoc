# CRAWLDoc: A Dataset for Robust Ranking of Bibliographic Documents

This repository contains code to reproduce the results from our paper **"CRAWLDoc: A Dataset for Robust Ranking of Bibliographic Documents"**, introducing:

1. **A Contextual Ranking Method (CRAWLDoc)** - Novel document-as-query approach for robust identification of bibliographic sources across web documents using:
   - Unified embeddings of content, URLs, and anchor texts
   - Layout-aware processing of HTML/PDF documents
   - Maximum Inner Product Search (MIPS) ranking

2. **A New Benchmark Dataset** - A comprehensive dataset for bibliographic source retrieval containing:
   - 600 publications from 6 major CS publishers (ACM, IEEE, Springer, etc.)
   - 72,483 annotated document relevancy labels
   - Complete bibliographic records with author affiliations
   - Publisher layout variations for robustness testing

## Key Features
- **Layout Independence**: Robust ranking across publisher website variations
- **Multi-Format Support**: Processes both HTML and PDF documents
- **One-Hop Context**: Evaluates linked resources within single crawl depth
- **Reproducible Baseline**: Includes pre-configured Jina Embeddings v2 model setup

## Structure
The repository is structured as follows:
- `dataset`: Contains the dataset with the bibliographic metadata and the linked websites
- `run_scripts`: Contains the scripts to train and test the models in a robustness check setup

## Dataset
To reproduce the results, use the dataset from the [dataset](./dataset) folder. The dataset is structured as follows:
```json
{
    "doi": "The DOI of the publication",
    "publisher_doi": "The DOI of the publisher",
    "publisher": "The publisher of the publication",
    "year": "The year of the publication",
    "title": "The title of the publication",
    "authors": [
        [
            "The name of the author",
            [
                "The affiliations of the author"
            ]
        ]
    ],
    "linked_websites": [
        {
            "id": "The id of the linked website",
            "anchor": "The anchor text of the linked website",
            "website": "The URL of the linked website",
            "label": "The label of the linked website"
        }
    ]
}
```

Due to legal reasons, we cannot provide the websites itself. However, in [/dataset](./dataset) you can find the scripts to crawl the websites.

## Experimental Setup
To reproduce the results, use the following python files:
- `train_retrieval.py` Train the retrieval models (Document and Query encoder) with the CrawlDoc procedure
- `eval_ranking.py` Evaluate the retrieval models

### Hyperparemeter search
The Hyperparameter search was conducted with [Weights and Biases](https://wandb.ai/). 
The config for the sweep is stored in `sweep.yaml`.