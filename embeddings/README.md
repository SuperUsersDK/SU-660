# Embeddings Examples (Python)

Undervisningsvenlige eksempler til embeddings i RAG-systemer.

Maal:

- forklare hvad embeddings er (og hvad de ikke er)
- vise forskel paa lexical vs dense retrieval
- vise hvorfor normalisering/afstandsmaal betyder noget
- demonstrere retrieval-strategier (metadata-filter, hybrid, multi-query)
- skabe et godt grundlag for RAG-design paa et 3-dages kursus

De fleste scripts kan koeres offline (kun Python standard library).
Kun demo 08 bruger OpenAI API (valgfrit).

```bash
python3 embeddings/01_lexical_vs_dense.py
python3 embeddings/02_similarity_metrics.py
python3 embeddings/03_query_embedding_strategies.py
python3 embeddings/04_chunk_vs_document_embeddings.py
python3 embeddings/05_metadata_filtering.py
python3 embeddings/06_hybrid_retrieval_rrf.py
python3 embeddings/07_model_tradeoffs.py
python3 embeddings/08_openai_embeddings_demo.py
```

Bemærk:

- `dense` i disse demoer er en didaktisk mock-embedding, ikke en rigtig model.
- Fokus er paa intuition, retrieval-adfaerd og designvalg.
- Brug demo 08 til at vise et rigtigt API-kald med et embedding-model endpoint.

