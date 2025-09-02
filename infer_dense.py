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
from sentence_transformers import SentenceTransformer


RETRIEVAL_MODEL = "sentence-transformers/contriever-sentencetransformer"
MODEL_PATH = "Llama-3.1-8B-Instruct"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _beta_fn(i):
    return 0.75

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
            scores[i] = scores[i] + beta * bm25_logits
        return scores


class BeirInfer:

    def __init__(self, data_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # self.model = LlamaForCausalLM.from_pretrained(MODEL_PATH).to("cuda").eval()
        self.model = LlamaForCausalLM.from_pretrained(MODEL_PATH).half().eval().to("cuda")
        self.retrieval_model = SentenceTransformer(RETRIEVAL_MODEL, device="cuda").eval()
        logging.info("Model loaded.")

        self.dataset = ir_datasets.load(data_dir)
        
        # get the documents
        self.document_token_list = []
        self.document_list = []
        self.doc_id_list = []
        for doc in self.dataset.docs_iter():
            token_list = self.tokenizer(doc.text, add_special_tokens=False).input_ids
            self.document_token_list.append(token_list)
            self.document_list.append(doc.text)
            self.doc_id_list.append(doc.doc_id)
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
    
    def prompt_fn(self, query):
        return f"Please write a scientific paper passage to support or refute the claim.\nClaim: {query}\nPassage:"
    
    def _get_text(self, query):
        prompt = self.prompt_fn(query)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # logits_processor_list = LogitsProcessorList([BM25Processor(self.bm25, _beta_fn, self.model.vocab_size)])
        generation_config = GenerationConfig(max_length=128,
                                             num_beams=1,
                                             do_sample=False,
                                             pad_token_id=128001)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            # logits_processor=logits_processor_list, 
                                            generation_config=generation_config)
        # print(generated_ids)  # Debugging line to print generated_ids
        generated_ids = generated_ids.cpu().tolist()  # Convert to numpy for BM25 scoring
        final_text = query + " " + self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt):]
        return final_text
        print(f"final text: {final_text}")
        query_embedding = self.retrieval_model.encode([final_text])
        scores = np.dot(passage_embedding, query_embedding[0]) # shape: (N, 1)
        scores = scores.flatten()
        # scores = self.bm25.bm25_score(generated_ids[0])
        return scores

    def _encode(self, text_list):
        batch_size = 128
        res = []
        for i in range(0, len(text_list), batch_size):
            res.append(self.retrieval_model.encode(text_list[i:i+batch_size], show_progress=False))
        return np.concatenate(res, axis=0)
    
    def search(self):
        # get the logits
        passage_embedding = self._encode(self.document_list)
        predicted_dict = {}
        query_list = []
        for query_id, query in tqdm(zip(self.query_id_list, self.query_list), desc="query rewrite", total=len(self.query_list)):
            query_list.append(self._get_text(query)) 
        query_embedding = self._encode(query_list)
        all_scores = query_embedding @ passage_embedding.T
        for i in range(len(query_list)):
            query_id = self.query_id_list[i]
            scores = all_scores[i]
            arg_sort = np.argsort(scores)
            predicted_dict[query_id] = {self.doc_id_list[i]: float(scores[i]) for i in arg_sort[-100:]}
        res = EvaluateRetrieval.evaluate(self.qrel_dict, predicted_dict, k_values=[10,100])
        return res



if __name__ == '__main__':
    nfcorpus = BeirInfer("beir/webis-touche2020")
    res = nfcorpus.search()
    print(res)

# nohup python infer.py > infer.log 2>&1 &