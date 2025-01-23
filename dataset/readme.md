# Dataset

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
