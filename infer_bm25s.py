import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import ir_datasets
import logging
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM, LogitsProcessorList, LogitsProcessor, GenerationConfig
from bm25 import BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

import Stemmer
import bm25s



MODEL_PATH = "Llama-3.1-8B-Instruct"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _beta_fn(i):
    return 1.0

def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with BEIR evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """

    results_record = [
        {"id": qid, "hits": results[i], "scores": list(scores[i])}
        for i, qid in enumerate(query_ids)
    ]

    result_dict_for_eval = {
        res["id"]: {
            docid: float(score) for docid, score in zip(res["hits"], res["scores"])
        }
        for res in results_record
    }

    return result_dict_for_eval

class BM25Processor(LogitsProcessor):

    def __init__(self, bm25, beta_fn, vocab_size):
        self.bm25 = bm25
        self.beta_fn = beta_fn
        self.vocab_size = vocab_size
    
    def __call__(self, input_ids, scores):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        Warning:
            input_ids contains the input prompt tokens and the generated tokens.
        """
        beta = self.beta_fn(input_ids.shape[1])
        if beta == 0:
            return scores
        for i in range(input_ids.shape[0]):
            bm25_logits = self.bm25.get_logits(input_ids[i].tolist(), self.vocab_size, 10)
            bm25_logits = torch.tensor(bm25_logits).to(scores.device)  
            # topk做个softmax？
            scores[i] = scores[i] + beta * bm25_logits
        return scores


class BeirInfer:

    def __init__(self, data_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # self.model = LlamaForCausalLM.from_pretrained(MODEL_PATH).to("cuda").eval()
        self.model = LlamaForCausalLM.from_pretrained(MODEL_PATH).half().eval().to("cuda")
        logging.info("Model loaded.")

        self.dataset = ir_datasets.load(data_dir)
        
        # get the documents
        self.document_token_list = []
        self.doc_id_list = []
        self.doc_text_list = []
        for doc in self.dataset.docs_iter():
            token_list = self.tokenizer(doc.text, add_special_tokens=False).input_ids
            self.document_token_list.append(token_list)
            self.doc_id_list.append(doc.doc_id)
            self.doc_text_list.append(doc.text)
        logging.info("Documents loaded.")

        # get the queries
        self.query_token_list = []
        self.query_list = []
        self.query_id_list = []
        for query in self.dataset.queries_iter():
            token_list = self.tokenizer(query.text, return_tensors="np", add_special_tokens=False).input_ids
            self.query_token_list.append(token_list)
            self.query_list.append(query.text)
            self.query_id_list.append(query.query_id)
        logging.info("Queries loaded.")

        # get the qrels
        self.qrel_dict = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in self.qrel_dict:
                self.qrel_dict[qrel.query_id] = {}
            self.qrel_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
        logging.info("Qrels loaded.")

        self.bm25 = BM25(self.document_token_list, k1=1.5, b=0.75, vocab_size=self.model.vocab_size, device=self.model.device)
        logging.info("BM25 loaded.")

        # search BM25 by bm25s
        self.stemmer = Stemmer.Stemmer('english')
        self.corpus_tokens_bm25s = bm25s.tokenize(
            self.doc_text_list, stemmer=self.stemmer, leave=False
        )
        self.bm25s_retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        self.bm25s_retriever.index(self.corpus_tokens_bm25s, leave_progress=False)
    
    def prompt_fn(self, query):
        return f"Please write a counter argument for the passage.\nPassage: {query}\nCounter Argument:"
    
    def _get_text(self, query):
        prompt = self.prompt_fn(query)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        logits_processor_list = LogitsProcessorList([BM25Processor(self.bm25, _beta_fn, self.model.vocab_size)])
        generation_config = GenerationConfig(max_length=2048,
                                             num_beams=1,
                                             do_sample=False,
                                             pad_token_id=128001)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            logits_processor=logits_processor_list, 
                                            generation_config=generation_config)
        # print(generated_ids)  # Debugging line to print generated_ids
        generated_ids = generated_ids.cpu().tolist()  # Convert to numpy for BM25 scoring
        final_text = query + " " + self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt):]
        return final_text
        scores = self.bm25.bm25_score(generated_ids[0])
        return scores
    

    def search(self):
        # get the logits
        query_list = []
        qids = []
        for query_id, query in tqdm(zip(self.query_id_list, self.query_list), desc="Searching", total=len(self.query_list)):
            query_list.append(self._get_text(query)) 
            # query_list.append(query) 
            qids.append(query_id)
        query_tokens = bm25s.tokenize(query_list, stemmer=self.stemmer, leave=False)
        ############## BENCHMARKING BEIR HERE ##############
        queried_results, queried_scores = self.bm25s_retriever.retrieve(query_tokens, corpus=self.doc_id_list, k=100)
        results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            self.qrel_dict, results_dict, [10, 100]
        )

        print(ndcg)
        print(recall)
        
        return ndcg, _map, recall, precision



if __name__ == '__main__':
    search_dataset = BeirInfer("beir/webis-touche2020")
    ndcg, _map, recall, precision = search_dataset.search()

