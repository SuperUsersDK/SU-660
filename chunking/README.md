# Chunking Examples (Python)

Denne mappe indeholder undervisningsvenlige Python-eksempler paa chunking-strategier.

Eksempel 01–07 bruger kun Python standard library. Eksempel 08 kræver LangChain
og tiktoken fra projektets dependencies. Kør fra repository-roden:

```bash
python3 chunking/01_fixed_size_words.py
python3 chunking/02_fixed_vs_overlap.py
python3 chunking/03_sentence_and_paragraph.py
python3 chunking/04_recursive_separator.py
python3 chunking/05_heading_aware_markdown.py
python3 chunking/06_semanticish_grouping.py
python3 chunking/07_parameter_grid_eval.py
python3 chunking/08_builtin_langchain_splitters.py
```

Formaal:

- vise chunk-graenser
- sammenligne retrieval-resultater
- demonstrere tradeoffs mellem strategier, chunk-stoerrelse og overlap
- sammenligne built-in chunking-strategier i LangChain

Bemærk:

- `08_builtin_langchain_splitters.py` kræver `langchain_text_splitters` (og `tiktoken` til `TokenTextSplitter`).
- Før demo 08, modul 2's token-chunking og chunking-opgaven køres
  `python tools/warmup_tokenizers.py` én gang med internetadgang.
  Token-data gemmes i `.cache/tiktoken/`, som de samme eksempler bruger offline.
  Se [fælles opsætning](../README.md) for cacheplacering og miljøvariabler.
- I de hjemmelavede recursive-eksempler begrænser `max_chars` basis-segmentet før
  overlap. Det færdige chunk kan være op til `max_chars + overlap_chars + 1` tegn
  langt inklusive mellemrum. LangChains egne splitters har deres egne størrelsesregler.
- Til undervisning findes et worksheet til demo 08: `chunking/08_builtin_langchain_splitters_worksheet.md`.
- Underviser-facit til worksheet: `chunking/08_builtin_langchain_splitters_worksheet_instructor.md`.
