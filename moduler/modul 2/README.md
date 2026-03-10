# Modul 2 Demoer

Denne mappe indeholder demoer og smaae eksempler til modul 2.

## RAG-demo uden aegte LLM

Filen `rag_simuleret_demo.py` viser forskellen mellem:

- en prompt uden RAG
- den samme prompt med simuleret RAG

RAG-delen er simuleret med lokal data direkte i appen. Der bruges ingen rigtig model, ingen embeddings og ingen vector database.

Formaalet er at vise den didaktiske pointe:

- uden RAG svarer systemet generelt og lidt loest
- med RAG bliver prompten grounded i konkret kontekst

Koer demoen med:

```bash
python "moduler/modul 2/rag_simuleret_demo.py"
```
