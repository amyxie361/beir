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

model_path = "splade/weights/distilsplade_max"
model = DRES(models.SPLADE(model_path), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="dot")
pickle.dump(retriever, open("splade-bioasq.retriever", 'wb'))
results = retriever.retrieve(corpus, queries)
pickle.dump(results, open("splade-bioasq.results",'wb'))

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(ndcg, _map, recall, precision)
