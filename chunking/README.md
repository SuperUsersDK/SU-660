# Chunking Examples (Python)

Denne mappe indeholder undervisningsvenlige Python-eksempler paa chunking-strategier.

Eksemplerne er lavet uden eksterne dependencies (kun Python standard library), saa de er nemme at koere:

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
- Til undervisning findes et worksheet til demo 08: `chunking/08_builtin_langchain_splitters_worksheet.md`.
- Underviser-facit til worksheet: `chunking/08_builtin_langchain_splitters_worksheet_instructor.md`.
