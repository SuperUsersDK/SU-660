# Chunking-eksempler (Python)

Mappen indeholder små, undervisningsvenlige Python-eksempler på chunking-strategier.
Eksempel 01–07 bruger kun Pythons standardbibliotek og den fælles `common.py`.
Eksempel 08 kræver LangChain og tiktoken fra projektets dependencies.

Kør fra repository-roden med `.venv` aktiveret:

```bash
python chunking/01_fixed_size_words.py
python chunking/02_fixed_vs_overlap.py
python chunking/03_sentence_and_paragraph.py
python chunking/04_recursive_separator.py
python chunking/05_heading_aware_markdown.py
python chunking/06_semanticish_grouping.py
python chunking/07_parameter_grid_eval.py
python chunking/08_builtin_langchain_splitters.py
```

| Eksempel | Viser |
|---|---|
| `01_fixed_size_words.py` | Fast antal ord pr. chunk |
| `02_fixed_vs_overlap.py` | Fast størrelse med og uden overlap |
| `03_sentence_and_paragraph.py` | Chunking ved sætninger og afsnit |
| `04_recursive_separator.py` | Rekursiv opdeling med separatorer |
| `05_heading_aware_markdown.py` | Chunking efter Markdown-overskrifter |
| `06_semanticish_grouping.py` | "Semantisk" gruppering af sætninger (tekstlig lighed) |
| `07_parameter_grid_eval.py` | Retrieval-kvalitet for forskellige chunk-størrelser og overlap |
| `08_builtin_langchain_splitters.py` | LangChains indbyggede splitters side om side |

Formål:

- vise chunk-grænser
- sammenligne retrieval-resultater
- demonstrere tradeoffs mellem strategier, chunk-størrelse og overlap
- sammenligne LangChains indbyggede chunking-strategier

Bemærk:

- `08_builtin_langchain_splitters.py` kræver `langchain_text_splitters` (og `tiktoken`
  til `TokenTextSplitter`).
- Før eksempel 08, modul 2's token-chunking og chunking-opgaven køres
  `python tools/warmup_tokenizers.py` én gang med internetadgang. Token-data gemmes i
  `.cache/tiktoken/`, som de samme eksempler bruger offline. Se
  [fælles opsætning](../README.md) for cacheplacering og miljøvariabler.
- I de hjemmelavede rekursive eksempler begrænser `max_chars` basis-segmentet før
  overlap. Den færdige chunk kan være op til `max_chars + overlap_chars + 1` tegn lang
  inklusive mellemrum. LangChains egne splitters har deres egne størrelsesregler.
- Worksheet til eksempel 08: `08_builtin_langchain_splitters_worksheet.md`.
  Underviser-facit: `08_builtin_langchain_splitters_worksheet_instructor.md`.
- Til en visuel, animeret sammenligning af strategierne med rigtige embeddings og
  Qdrant, se [chunking-laboratoriet](../demoer/modul%202/chunking_lab/README.md).
