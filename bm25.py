import math
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SubDocuments:
    def __init__(self, sub_tf_idf_table_doc, device):
        """
        Initialize the SubDocuments class with a subset of the TF-IDF table and a specified device.

        Args:
            sub_tf_idf_table_doc (list of dict): subset of tf_idf_table_doc, sub_tf_idf_table_doc[doc_id][token] = tf_idf_score
            device (str): The device (e.g., 'cpu' or 'cuda') where the tensors will be stored and computed.

        This method performs the following steps:
        1. Maps each unique token to a unique index and stores the mapping in `self.token2idx`.
           This is to narrow the tokens to a manageable set of indices, speed up the computation and reduce memory usage.
        2. Creates a tensor `self.idx2token` that maps indices back to their corresponding tokens.
        3. Creates a tensor `self.table` that stores the TF-IDF scores for each document and new IDs.
        """
        self.token2idx = {}
        self.idx2token = []
        for doc in sub_tf_idf_table_doc:
            for token, tf_idf in doc.items():
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.idx2token)
                    self.idx2token.append(token)
        self.idx2token = torch.tensor(self.idx2token, device=device)
        self.table = torch.zeros((len(sub_tf_idf_table_doc), len(self.token2idx)), device=device)
        for i, doc in enumerate(sub_tf_idf_table_doc):
            for token, tf_idf in doc.items():
                self.table[i, self.token2idx[token]] = tf_idf

    def get_token_size(self):
        return len(self.token2idx)
    
    def get_logits(self, query, vocab_size, top_k):
        """
        Calculate the logits for a given query based on the sub documents.
        
        Args:
            query (list of int): The query represented as a list of token IDs.
            vocab_size (int): The size of the LLM vocabulary. This is used to create a return tensor of the appropriate size.
            top_k (int): The number of top scoring tokens to consider when calculating the logits.

        Returns:
            torch.Tensor: A tensor of shape (top_k, vocab_size) containing the logits for each document.
        """
        # 1. remove tokens not in the vocabulary and convert token IDs to new IDs
        query = [self.token2idx[token] for token in query if token in self.token2idx]
        query = torch.tensor(query, device=self.table.device)
        # 2. calculate the BM25 score for each document
        doc_score = self.table[:, query].sum(dim=1, keepdim=True)
        
        # 3. calculate the logits for each document
        logits = self.table + doc_score
        # 4. get the top k documents for every token
        logits = torch.topk(logits, top_k, dim=0).values

        # 5. convert new IDs back to token IDs and fill in the logits tensor
        ret = torch.zeros((logits.shape[0], vocab_size), device=self.table.device)
        ret[:, self.idx2token] = logits
        return ret


class BM25:
    def __init__(self, documents, k1, b, vocab_size, batch_size=800, device='cpu'):
        """
        Initialize the BM25 model with a list of documents and parameters k1 and b.

        Args:
            documents (list of list of int): A list of tokenized documents, where each document is represented as a list of token IDs.
            k1 (float): A parameter that controls the non-linear term in the IDF formula.
            b (float): A parameter that controls the scaling of the document length normalization.
            vocab_size (int): The size of the vocabulary. 
            batch_size (int): The number of documents to process in each batch. 
            device (str): The device to use for computation.

        Example:
            >>> documents = [[12, 100, 12], [100, 12, 100], [100, 100, 100]]
            >>> bm25 = BM25(documents, k1=1.5, b=0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.token_doc_freq = {} # token_doc_freq[token_id] means the number of documents that contain token_id
        self.avg_doc_len = sum([len(doc) for doc in documents]) / len(documents)
        self.tf_idf_table_doc = [{} for _ in range(len(documents))] # tf_idf_table_doc[doc_id][token_id] means the BM25 score(tf*idf value) of token_id in doc_id
        self.token2doc = [[] for _ in range(vocab_size)] # token2doc[token_id] means the list of doc_id that contains token_id
        self.all_token_list=[] # all different token_id in all documents
        self._calculate_doc_freq()
        self._calculate_tf_idf_table_doc()

        self.sub_documents_list = [] # sub_documents_list[i] means the i-th sub-documents, used for efficient logits calculation
        for i in range(0, len(self.documents), batch_size):
            self.sub_documents_list.append(SubDocuments(self.tf_idf_table_doc[i:i+batch_size], device=device))
        avg_sub_token_size = sum([sub_doc.get_token_size() for sub_doc in self.sub_documents_list]) / len(self.sub_documents_list)
        logging.info(f"BM25 initialized with k1={k1}, b={b}, avg_doc_len={self.avg_doc_len:.2f}, token_size={len(self.all_token_list)}, avg_sub_token_size={avg_sub_token_size:.2f}")
        
    def _calculate_doc_freq(self):
        for doc_idx, doc in enumerate(self.documents):
            unique_tokens = set(doc)
            for token in unique_tokens:
                if token not in self.token_doc_freq:
                    self.token_doc_freq[token] = 0
                self.token_doc_freq[token] += 1
                self.token2doc[token].append(doc_idx) # Store the document index where the token appears
        self.all_token_list = list(self.token_doc_freq.keys())
    
    def _idf(self, token):
        N = len(self.documents)
        df = self.token_doc_freq.get(token, 0)
        if df == 0:
            return 0
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def _calculate_tf_idf_table_doc(self):
        for doc_idx, doc in enumerate(self.documents):
            tf_idf_values = {}
            for token in set(doc):
                f = doc.count(token)
                idf = self._idf(token)
                k1, b = self.k1, self.b
                tf = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (len(doc) / self.avg_doc_len)))
                tf_idf_values[token] = tf*idf
            self.tf_idf_table_doc[doc_idx] = tf_idf_values

    def bm25_score(self, query):
        """
        Calculate the BM25 score for a given query against all documents.
        Args:
            query (list of int): The query represented as a list of token IDs.
            
        Returns:
            Numpy array of BM25 scores for each document.
        """
        scores = np.zeros(len(self.documents))
        for token in query:
            for doc_idx in self.token2doc[token]:
                scores[doc_idx] += self.tf_idf_table_doc[doc_idx][token]
        
        return scores
    
    def get_logits(self, query, vocab_size, top_k):
        """
        Get the logits for a given query.
        Args:
            query (list of int): The query represented as a list of token IDs.
            vocab_size (int): The size of the vocabulary. Not used in BM25 calculation but included for consistency with other models.
            topk (int): The number of top scoring tokens to consider. Not used in BM25 calculation but included for consistency with other models.

        Returns:
            Numpy array of logits of length vocab_size, where each element is the BM25 score for the corresponding token in the vocabulary.
        """
        logits = []
        for sub_doc in self.sub_documents_list:
            logits.append(sub_doc.get_logits(query, vocab_size, top_k))
            if len(logits) >= 100:
                logits = torch.concat(logits, dim=0)
                logits = [torch.topk(logits, top_k, dim=0).values]
        logits = torch.concat(logits, dim=0)
        logits = torch.topk(logits, top_k, dim=0).values
        logits = torch.mean(logits, dim=0)
        logits = torch.softmax(logits, dim=0)
        return logits



