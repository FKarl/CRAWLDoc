import json
import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score

from train_retrieval import embed_text, batch_embed_text, similarity_dot_product, info_nce_loss
from transformers import AutoModel, AutoTokenizer, AutoConfig
import wandb


def evaluate(args: argparse.Namespace, query_model: PreTrainedModel, document_model: PreTrainedModel,
             tokenizer: PreTrainedTokenizer, data: List[Tuple[str, List[Tuple[str, int]]]], loss_function):
    """
    Evaluate the model.
    @param args: argparse.Namespace: The arguments. Including training args like batch size, epochs, learning rate, etc.
    @param query_model: transformers.PreTrainedModel: The model to use for the query.
    @param document_model: transformers.PreTrainedModel: The model to use for the documents.
    @param tokenizer: transformers.PreTrainedTokenizer: The tokenizer to use for the models.
    @param data: List[Tuple[str, int]]: The data to evaluate. Each tuple contains a text and a label.
    @ return: None
    """
    logging.info("Evaluating the model.")
    # Set the models to evaluation mode
    query_model.eval()
    document_model.eval()

    # Initialize metrics
    mrr = 0
    ndcg = 0
    map = 0
    precision_at_k = [0] * args.k_range[1]
    recall_at_k = [0] * args.k_range[1]

    # "10.1007" "10.1016" "10.1109" "10.1145" "10.3390" "10.48550"
    mrr_per_publisher = {
        "10.1007": 0,
        "10.1016": 0,
        "10.1109": 0,
        "10.1145": 0,
        "10.3390": 0,
        "10.48550": 0
    }
    ndcg_per_publisher = {
        "10.1007": 0,
        "10.1016": 0,
        "10.1109": 0,
        "10.1145": 0,
        "10.3390": 0,
        "10.48550": 0
    }
    map_per_publisher = {
        "10.1007": 0,
        "10.1016": 0,
        "10.1109": 0,
        "10.1145": 0,
        "10.3390": 0,
        "10.48550": 0
    }
    publisher_count = {
        "10.1007": 0,
        "10.1016": 0,
        "10.1109": 0,
        "10.1145": 0,
        "10.3390": 0,
        "10.48550": 0
    }

    # Evaluate the model
    batch_size = args.eval_batch_size
    context_size = args.max_context
    test_loss = 0
    for i, (query_text, search_texts, publisher_doi) in enumerate(tqdm(data, desc="Evaluating Query")):
        query_embedding = embed_text(query_text, query_model, tokenizer, max_length=context_size)
        positive_search_texts = [text for text, label in search_texts if label == 1]
        negative_search_texts = [text for text, label in search_texts if label == 0]

        # embedd all documents (in batches as well because of memory constraints)
        positive_embeddings = torch.tensor([])
        for j in range(0, len(positive_search_texts), batch_size):
            positive_batch = positive_search_texts[j:j + batch_size]
            batch_emb = batch_embed_text(positive_batch, document_model, tokenizer, max_length=context_size)
            batch_emb = batch_emb.detach().cpu()
            positive_embeddings = torch.cat(
                (positive_embeddings, batch_emb), 0)

        negative_embeddings = torch.tensor([])
        for j in range(0, len(negative_search_texts), batch_size):
            negative_batch = negative_search_texts[j:j + batch_size]
            batch_emb = batch_embed_text(negative_batch, document_model, tokenizer, max_length=context_size)
            batch_emb = batch_emb.detach().cpu()
            negative_embeddings = torch.cat(
                (negative_embeddings, batch_emb), 0)

        for j in range(0, len(positive_search_texts), batch_size):
            positive_batch = positive_embeddings[j:j + batch_size]
            negative_batch = negative_embeddings[j:j + batch_size * args.train_neg_examples]

            # move to device
            positive_batch = positive_batch.to(document_model.device)
            negative_batch = negative_batch.to(document_model.device)

            batched_query_embedding = query_embedding.repeat(len(positive_batch), 1)

            loss = loss_function(batched_query_embedding, positive_batch, negative_batch)
            test_loss += loss.item()

            # freeup memory
            del positive_batch
            del negative_batch
            del batched_query_embedding
            del loss

        # Compute metrics
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings])
        scores = []  # similarity_dot_product(query_embedding, score.to(query_model.device)).item() for score in all_embeddings
        query_embedding = query_embedding.squeeze()
        for score in all_embeddings:
            score = score.to(query_embedding.device)
            scores.append(similarity_dot_product(query_embedding, score).item())
            score = score.detach().cpu()

        labels = torch.cat([torch.ones(len(positive_search_texts)), torch.zeros(len(negative_search_texts))]).tolist()
        score_labels = list(zip(scores, labels))
        ranked_score_labels = sorted(score_labels, key=lambda x: x[0], reverse=True)

        scores, labels = zip(*ranked_score_labels)

        # Calculate precision@k and recall@k
        relevant_docs = sum(labels)
        for k in range(1, args.k_range[1] + 1):
            if k <= len(labels):
                retrieved_docs = labels[:k]
                relevant_retrieved_docs = sum(retrieved_docs)
                precision_at_k[k - 1] += relevant_retrieved_docs / k
                recall_at_k[k - 1] += relevant_retrieved_docs / relevant_docs if relevant_docs > 0 else 0

        # calc Mean Reciprocal Rank
        for i, label in enumerate(labels):
            if label == 1:
                value = 1 / (i + 1)
                mrr += value
                mrr_per_publisher[publisher_doi] += value
                break

        # Reshape labels and scores to be 2D array-like
        labels = np.array(labels).reshape(1, -1)
        scores = np.array(scores).reshape(1, -1)

        ndcg_res = ndcg_score(labels, scores)
        ndcg += ndcg_res
        ndcg_per_publisher[publisher_doi] += ndcg_res
        map_res = label_ranking_average_precision_score(labels, scores)
        map += map_res
        map_per_publisher[publisher_doi] += map_res

        publisher_count[publisher_doi] += 1

        # freeup memory
        del positive_embeddings
        del negative_embeddings
        del all_embeddings
        del scores
        del labels
        del score_labels
        del ranked_score_labels

    test_loss /= len(data)
    mrr /= len(data)
    ndcg /= len(data)
    map /= len(data)

    for publisher in publisher_count:
        if publisher_count[publisher] > 0:
            mrr_per_publisher[publisher] /= publisher_count[publisher]
            ndcg_per_publisher[publisher] /= publisher_count[publisher]
            map_per_publisher[publisher] /= publisher_count[publisher]

            wandb.log(
                {f"MRR_{publisher}": mrr_per_publisher[publisher], f"NDCG_{publisher}": ndcg_per_publisher[publisher],
                 f"MAP_{publisher}": map_per_publisher[publisher]})

    logging.info(f"Validation Loss: {test_loss}")
    logging.info(f"Mean Reciprocal Rank: {mrr}")
    logging.info(f"Normalized Discounted Cumulative Gain: {ndcg}")
    logging.info(f"Mean Average Precision: {map}")

    logging.info(f"Mean Reciprocal Rank per publisher: {mrr_per_publisher}")
    logging.info(f"Normalized Discounted Cumulative Gain per publisher: {ndcg_per_publisher}")
    logging.info(f"Mean Average Precision per publisher: {map_per_publisher}")

    # Average precision@k and recall@k over all queries
    precision_at_k = [p / len(data) for p in precision_at_k]
    recall_at_k = [r / len(data) for r in recall_at_k]

    logging.info(f"Precision@k_list: {precision_at_k}")
    logging.info(f"Recall@k_list: {recall_at_k}")

    for k in range(1, args.k_range[1] + 1):
        wandb.log({"Precision@k": precision_at_k[k - 1], "Recall@k": recall_at_k[k - 1]}, step=k)

    wandb.log(
        {"Validation Loss": test_loss, "Mean Reciprocal Rank": mrr, "Normalized Discounted Cumulative Gain": ndcg,
         "Mean Average Precision": map, "Precision@_list": precision_at_k, "Recall@k_list": recall_at_k})

    return test_loss, mrr, ndcg, map, precision_at_k, recall_at_k


def load_data(data_path: str) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """
    Load the metadata from the json files, then loads the according documents.
    @param data_path: str: The path to the json file containing the metadata.
    @return: Tuple[List[Tuple[str, List[Tuple[str, int]]], List[Tuple[str, List[Tuple[str, int]]]]]: The training and validation data.
    """
    with open(data_path, "r") as f:
        train_meta = json.load(f)
    test_data = []

    for paper in train_meta:

        doi = paper['doi']
        publisher = paper['publisher_doi']

        with open(f"websites/{doi}/website.json", "r") as f:
            query = json.load(f)
        # convert query to string
        query = json.dumps(query)

        linked_websites = paper['linked_websites']
        documents = []
        for website in linked_websites:
            label = website['label']
            id = website['id']
            anchor = website['anchor']
            link = website['website']

            try:
                with open(f"websites/{doi}/site_{id}/website.json", "r") as f:
                    doc = json.load(f)
            except FileNotFoundError:
                # skip if it does not exist (scraping error) # Could also be removed from the json file. But keep it for now to make sure the data is correct
                continue
            # check if query and doc are the same
            if query == json.dumps(doc):
                continue
            # build input "prompt"
            prompt = f"{anchor} [SEP] {link} [SEP] {json.dumps(doc)}"

            documents.append((prompt, label))

        test_data.append((query, documents, publisher))

    return test_data


if __name__ == '__main__':
    """
    Evaluate the ranking model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_model", type=str, default="./models/optimized_hp/query_model",
                        help="The model to use for the query.")
    parser.add_argument("--document_model", type=str, default="./models/optimized_hp/document_model",
                        help="The model to use for the documents.")
    parser.add_argument("--tokenizer", type=str, default="./models/optimized_hp/document_model",
                        help="The tokenizer to use for the models.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="The batch size for evaluation.")
    parser.add_argument("--max_context", type=int, default=2048, help="The maximum context size.")
    parser.add_argument("--log_level", type=str, default="INFO", help="The log level.")
    parser.add_argument("--run_name", type=str, default="test", help="The name of the run.")
    parser.add_argument("--train_neg_examples", type=int, default=5,
                        help="The number of negative examples per positive example.")
    parser.add_argument("--k_range", type=int, nargs=2, default=[1, 20], help="The range of k for the evaluation.")
    parser.add_argument("--test_data", type=str, default="dataset/test.json", help="The path to the test data.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # wandb init
    wandb.init(project="Paper_IR", name=args.run_name, config=args)

    # Load the models
    query_model = AutoModel.from_pretrained(args.query_model, trust_remote_code=True)
    document_model = AutoModel.from_pretrained(args.document_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model.to(device)
    document_model.to(device)
    # set to eval mode
    query_model.eval()
    document_model.eval()

    # Load the data
    test_data = load_data(args.test_data)

    # test
    evaluate(args, query_model, document_model, tokenizer, test_data, info_nce_loss)
