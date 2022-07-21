from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import pickle

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()])

data_path = './datasets/bioasq'

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=256)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
pickle.dump(retriever, open("sbert-bioasq.retriever", 'wb'))
results = retriever.retrieve(corpus, queries)
pickle.dump(results, open("sbert-bioasq.results",'wb'))

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(ndcg, _map, recall, precision)
