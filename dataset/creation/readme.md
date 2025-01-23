# Dataset

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Structure](#structure)
- [Creation](#creation)
- [Helper](#helper)

## Introduction
This dataset is designed for the task of retrieving bibliographic metadata sources from heterogeneous web sources. It consists of 600 publications from 6 publishers and is divided into various splits for training, testing, and validation.

## Data
The dataset is stored in the following JSON files:

- `full_dataset.json`: The main dataset
- `train.json`: The training split of the dataset (80%)
- `test.json`: The test split of the dataset (10%)
- `validation.json`: The validation split of the dataset (10%)

For our robustness analysis, we created the following splits:
- `test_10.1007.json`: A split of the dataset with only 10.1007 publications
- `train_all_except_10.1007.json`: A split of the dataset with all publications except 10.1007

- `test_10.1016.json`: A split of the dataset with only 10.1016 publications
- `train_all_except_10.1016.json`: A split of the dataset with all publications except 10.1016

- `test_10.1109.json`: A split of the dataset with only 10.1109 publications
- `train_all_except_10.1109.json`: A split of the dataset with all publications except 10.1109

- `test_10.1145.json`: A split of the dataset with only 10.1145 publications
- `train_all_except_10.1145.json`: A split of the dataset with all publications except 10.1145

- `test_10.3390.json`: A split of the dataset with only 10.3390 publications
- `train_all_except_10.3390.json`: A split of the dataset with all publications except 10.3390

- `test_10.48550.json`: A split of the dataset with only 10.48550 publications
- `train_all_except_10.48550.json`: A split of the dataset with all publications except 10.48550

## Structure
The dataset is structured as follows:
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

## Creation
The following scripts are used to create the dataset in the order they are listed. Note that steps 1, 2, and 5 are not needed for the reproduction of the results, as the dataset is already created and stored in the above-listed JSON files.

1. `create_dataset.py`: Create the dataset based on the [DBLP dump](https://dblp.uni-trier.de/xml/)
2. `create_dataset_splits.py`: Create the splits of the dataset
3. `scrape_web_documents.py`: Scrape the web documents by emulating a browser
4. `postprocess_bounding_boxes.py`: Postprocess the bounding boxes of the scraped web documents to round them to the nearest multiple of 5
5. `labeling_tool.py`: Label the scraped web documents with a GUI

## Helper
The following JSON files are helper data for the dataset:
- `anchor_rules.json`: The rules for the anchor text
- `anchor_whitelist.json`: The whitelist for the anchor text
- `website_rules.json`: The rules for the website
- `website_whitelist.json`: The whitelist for the website
- `html_rules.json`: The rules for the HTML files
- `empty.json`: An empty JSON array
- `publisher.json`: A list of all publishers on DBLP with their count of publications

The following scripts are used as helpers for small tasks:
- `check_data.py`: Compare our data with the CrossRef API
- `dataset_statistics.py`: Create plots and statistics about the dataset
- `create_html_link_rules.py`: Create the HTML link rules for the dataset based on regex
- `calculate_rule_coverage.py`: Calculate the coverage of the rules on the dataset