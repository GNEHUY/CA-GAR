# CA-GAR: Context-Aware Alignment of LLM Generation for Document Retrieval


**This paper has been accepted at the ACL 2025 Findings.** [Paper](https://aclanthology.org/2025.findings-acl.303.pdf)

we introduce a novel approach called Context-Aware GenerationAugmented Retrieval (CA-GAR), which incorporates corpus information into the generation process of LLMs. Specifically, at the core of our approach is the optimization of the model’s autoregressive generation process by leveraging relevant
document information from the corpus to influence
token selection. To achieve this, we propose a Distribution Alignment Strategy, which utilizes a
lexicon-based method to extract corpus information. This strategy approximates the optimization
of the model’s autoregressive generation process,
ensuring that the generated content is better aligned
with the target document corpus.

## Citation

```
@inproceedings{yu-etal-2025-ca,
    title = "{CA}-{GAR}: Context-Aware Alignment of {LLM} Generation for Document Retrieval",
    author = "Yu, Heng  and
      Kang, Junfeng  and
      Li, Rui  and
      Liu, Qi  and
      He, Liyang  and
      Huang, Zhenya  and
      Shen, Shuanghong  and
      Lu, Junyu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.303/",
    doi = "10.18653/v1/2025.findings-acl.303",
    pages = "5836--5849",
    ISBN = "979-8-89176-256-5",
}
```