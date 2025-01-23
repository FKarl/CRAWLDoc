from typing import List, Tuple

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from numpy.linalg import norm
import torch
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE

from tqdm import tqdm, trange
from info_nce import InfoNCE
from torch.nn import TripletMarginLoss
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW

import argparse
import logging
import os
import json
import random

from torch.cuda.amp import GradScaler, autocast
import wandb
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score


def similarity_cosine(v1: torch.Tensor, v2: torch.Tensor):
    """Compute the cosine similarity between two vectors.
    @param v1: (torch.Tensor): A vector.
    @param v2: (torch.Tensor): Another vector.
    @return:  torch.Tensor: The cosine similarity between the two vectors.

    """
    return torch.cosine_similarity(v1, v2, dim=0)


def similarity_dot_product(v1: torch.Tensor, v2: torch.Tensor):
    """Compute the dot product between two vectors.
    @param v1: (torch.Tensor): A vector.
    @param v2: (torch.Tensor): Another vector.
    @return:  torch.Tensor: The dot product between the two vectors.
    """
    return torch.dot(v1, v2)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor):
    """Compute the InfoNCE loss.
    @param anchor: torch.Tensor: The anchor sample.
    @param positive: torch.Tensor: The positive samples.
    @param negatives: torch.Tensor: The negative samples.
    @param temperature: float: The temperature for the softmax.
    @return: torch.Tensor: The InfoNCE loss.
    """
    logging.debug("Computing the InfoNCE loss.")
    return InfoNCE()(anchor, positive, negatives)


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor):
    """Compute the triplet loss.
    @param anchor: torch.Tensor: The anchor sample.
    @param positive: torch.Tensor: The positive sample.
    @param negatives: torch.Tensor: The negative samples.
    @param margin: float: The margin for the triplet loss.
    @return: torch.Tensor: The triplet loss.
    """
    logging.debug("Computing the triplet loss.")
    return TripletMarginLoss()(anchor.unsqueeze(0), positive.unsqueeze(0), negatives.unsqueeze(0))


def embed_text(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length=8192):
    """Embeds text using the provided model and tokenizer.
    @param text: str: The text to embed.
    @param model: transformers.PreTrainedModel: The model to use for embedding.
    @param tokenizer: transformers.PreTrainedTokenizer: The tokenizer to use for embedding.
    @return: torch.Tensor: The (pooled) embeddings of the text.
    """
    input_ids = tokenizer.encode(text,
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length
                                 ).to(model.device)
    embeddings = model(input_ids)

    return embeddings.pooler_output


def batch_embed_text(texts: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length=8192):
    """Embeds text using the provided model and tokenizer.
    @param texts: List[str]: The texts to embed.
    @param model: transformers.PreTrainedModel: The model to use for embedding.
    @param tokenizer: transformers.PreTrainedTokenizer: The tokenizer to use for embedding.
    @return: torch.Tensor: The (pooled) embeddings of the text.
    """
    encoding = tokenizer.batch_encode_plus(texts,
                                           return_tensors="pt",
                                           padding=True,
                                           truncation=True,
                                           max_length=max_length
                                           ).to(model.device)

    embeddings = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'],
                       token_type_ids=encoding['token_type_ids'])

    return embeddings.pooler_output


def vector_search(query_embedding, search_embeddings):
    """Searches for the most relevant search document.
    @param query_embedding: torch.Tensor: The embedding of the query document.
    @param search_embeddings: List[Tuple[torch.Tensor, int]]: The embeddings of the search documents.
    @return: a List of documents sorted by relevance.
    """
    similarity = similarity_dot_product
    search_embeddings.sort(key=lambda x: similarity(query_embedding, x[0]), reverse=True)
    return search_embeddings


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
    # if the data is empty, return
    if data is None or len(data) == 0:
        logging.warning("No data to evaluate.")
        return -1, -1, -1, -1

    logging.info("Evaluating the model.")
    # Set the models to evaluation mode
    query_model.eval()
    document_model.eval()

    # Initialize metrics
    mrr = 0
    ndcg = 0
    map = 0

    # Evaluate the model
    batch_size = args.eval_batch_size
    context_size = args.max_context
    val_loss = 0
    for i, (query_text, search_texts) in enumerate(tqdm(data, desc="Evaluating Query")):
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
            val_loss += loss.item()

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
        # log the rankings in wandb
        wandb.log({"Rankings": ranked_score_labels})

        scores, labels = zip(*ranked_score_labels)

        # calc Mean Reciprocal Rank
        for i, label in enumerate(labels):
            if label == 1:
                mrr += 1 / (i + 1)
                break

        # Reshape labels and scores to be 2D array-like
        labels = np.array(labels).reshape(1, -1)
        scores = np.array(scores).reshape(1, -1)

        ndcg += ndcg_score(labels, scores)
        map += label_ranking_average_precision_score(labels, scores)

        # freeup memory
        del positive_embeddings
        del negative_embeddings
        del all_embeddings
        del scores
        del labels
        del score_labels
        del ranked_score_labels

    val_loss /= len(data)
    mrr /= len(data)
    ndcg /= len(data)
    map /= len(data)

    logging.info(f"Validation Loss: {val_loss}")
    logging.info(f"Mean Reciprocal Rank: {mrr}")
    logging.info(f"Normalized Discounted Cumulative Gain: {ndcg}")
    logging.info(f"Mean Average Precision: {map}")

    wandb.log(
        {"Validation Loss": val_loss, "Mean Reciprocal Rank": mrr, "Normalized Discounted Cumulative Gain": ndcg,
         "Mean Average Precision": map})

    query_model.train()
    document_model.train()

    return val_loss, mrr, ndcg, map


def train(args: argparse.Namespace, query_model: PreTrainedModel, document_model: PreTrainedModel,
          tokenizer: PreTrainedTokenizer, train_data: List[Tuple[str, List[Tuple[str, int]]]],
          val_data: List[Tuple[str, List[Tuple[str, int]]]]):
    """
    Train the model with the provided data and contrastive loss function.
    """
    logging.info("Training the model.")
    query_model.train()
    document_model.train()
    optimizer = AdamW(list(query_model.parameters()) + list(document_model.parameters()), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    loss_function = LOSS_FUNCTIONS[args.loss_function]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model.to(device)
    document_model.to(device)

    context_size = args.max_context

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Initialize early stopping variables
    best_metric = float('inf')  # Change to -float('inf') if a higher metric is better
    patience_counter = 0

    for epoch in trange(args.epochs, desc="Epoch"):
        random.shuffle(train_data)
        epoch_loss = 0
        for i, (query_text, search_texts) in enumerate(tqdm(train_data, desc="Training Query")):
            query_loss = 0
            positive_search_texts = [text for text, label in search_texts if label == 1]
            negative_search_texts = [text for text, label in search_texts if label == 0]
            random.shuffle(negative_search_texts)
            random.shuffle(positive_search_texts)
            batch_size = args.train_batch_size
            for j in range(0, len(positive_search_texts), batch_size):
                # Use autocast and GradScaler for mixed precision training
                with autocast():
                    # we need to encode the query every time again as the model is updated inbetween
                    query_embedding = embed_text(query_text, query_model, tokenizer, max_length=context_size)

                    positive_batch = positive_search_texts[j:j + batch_size]
                    negative_batch = negative_search_texts[
                                     j * args.train_neg_examples:j + batch_size * args.train_neg_examples]

                    # check if there are still samples left (can be assumed, but still check it)
                    if len(positive_batch) == 0 or len(negative_batch) == 0:
                        break

                    positive_embeddings = batch_embed_text(positive_batch, document_model, tokenizer,
                                                           max_length=context_size)
                    negative_embeddings = batch_embed_text(negative_batch, document_model, tokenizer,
                                                           max_length=context_size)

                    batched_query_embedding = query_embedding.repeat(len(positive_batch), 1)

                    logging.debug(
                        f"embedding dimensions: Q: {batched_query_embedding.shape}, P: {positive_embeddings.shape}, N: {negative_embeddings.shape}")

                    loss = loss_function(batched_query_embedding, positive_embeddings, negative_embeddings)
                    query_loss += loss.item()
                    wandb.log({"Batch Loss": loss.item()})

                # backpropagation
                scaler.scale(loss).backward()
                if i % args.num_accumulation_steps == 0 or i == len(train_data) - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # freeup memory
                del positive_embeddings
                del negative_embeddings
                del batched_query_embedding
                del loss

            query_loss /= len(positive_search_texts)
            epoch_loss += query_loss
            logging.debug(f"Query {i}, Loss: {query_loss}")
            wandb.log({"Query": i, "Query Loss": query_loss})

        epoch_loss /= len(train_data)
        logging.debug(f"Epoch {epoch}, Loss: {epoch_loss}")
        wandb.log({"Epoch": epoch, "Epoch Loss": epoch_loss})

        current_eval_metric, _, _, _ = evaluate(args, query_model, document_model, tokenizer, val_data, loss_function)

        if not args.disable_early_stopping:
            # Check for improvement
            if current_eval_metric < best_metric:  # Change to > if a higher metric is better
                best_metric = current_eval_metric
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= args.patience:
                print("Early stopping due to no improvement in validation metric.")
                break

    # save the models
    save_path = f"{args.save_dir}/{args.run_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    query_model.save_pretrained(f"{save_path}/query_model")
    document_model.save_pretrained(f"{save_path}/document_model")
    # also save the tokenizer in the same folders
    tokenizer.save_pretrained(f"{save_path}/query_model")
    tokenizer.save_pretrained(f"{save_path}/document_model")


def load_data(train_data_path: str, val_data_path: str):
    """
    Load the metadata from the json files, then loads the according documents.
    @param train_data_path: str: The path to the training data.
    @param val_data_path: str: The path to the validation data.
    @return: Tuple[List[Tuple[str, List[Tuple[str, int]]], List[Tuple[str, List[Tuple[str, int]]]]]: The training and validation data.
    """
    with open(train_data_path, "r") as f:
        train_meta = json.load(f)
    train_data = []

    for paper in train_meta:

        doi = paper['doi']

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

            # build input "prompt"
            prompt = f"{anchor} [SEP] {link} [SEP] {json.dumps(doc)}"

            documents.append((prompt, label))

        train_data.append((query, documents))

    # now the same for the validation data
    with open(val_data_path, "r") as f:
        val_meta = json.load(f)
    val_data = []

    for paper in val_meta:

        doi = paper['doi']

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

        val_data.append((query, documents))

    return train_data, val_data


LOSS_FUNCTIONS = {
    'info_nce': info_nce_loss,
    'triplet': triplet_loss
}

if __name__ == '__main__':
    """
    Train the retrieval model.
    """
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--chache_dir', type=str, default='./tmp')
    # Training args
    parser.add_argument('--train_batch_size', type=int, default=2,
                        help='Batch size for training, should be >= 2 as 1 is for the positive example')
    parser.add_argument('--train_neg_examples', type=int, default=5,
                        help='Number of negative examples for each positive example')
    parser.add_argument('--max_context', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--loss_function', choices=LOSS_FUNCTIONS.keys(), default='info_nce')
    parser.add_argument('--num_accumulation_steps', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--run_name', type=str, default='default_run')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait for improvement before stopping.')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--train_data', type=str, default='dataset/train.json')
    parser.add_argument('--val_data', type=str, default='dataset/validation.json')
    args = parser.parse_args()

    # set logging level
    logging.basicConfig(level=args.log_level)

    # wandb init
    wandb.init(project="Paper_IR", name=args.run_name, config=args)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.debug(f'args: {args}')

    # load the data
    train_data, val_data = load_data(args.train_data, args.val_data)

    logging.info("Loading the models and tokenizer.")
    query_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", cache_dir=args.chache_dir,
                                            trust_remote_code=True)
    document_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", cache_dir=args.chache_dir,
                                               trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

    # Log model parameters and gradients
    wandb.watch(query_model)
    wandb.watch(document_model)

    train(args, query_model, document_model, tokenizer, train_data, val_data)
