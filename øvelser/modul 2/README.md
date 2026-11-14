# Øvelser til modul 2

Mappen har to øvelser med skabeloner til kursisterne og løsninger til underviseren.
Kør fra repository-roden med `.venv` aktiveret.

| Øvelse | Skabelon | Løsning |
|---|---|---|
| Filtrér brugbare dokumenter til RAG | `filter_template.py` | `løsning/filter_losning.py` |
| Chunking med tokens og struktur | `chunking_template.py` | `løsning/chunking_løsning.py` |

## Opgave 1: Filtrér brugbare dokumenter til RAG

Deltagerne arbejder med et lille dokumentsæt af blandet kvalitet i `test_dokumenter/`
(14 filer). Målet er et filter, der kun bevarer dokumenter, som er gode nok til at
blive brugt i et RAG-system.

Dokumentsættet indeholder med vilje en blanding af:

- rigtige faglige tekster
- navigationstekst
- kladder
- dubletter eller næsten-dubletter
- korte irrelevante tekster

### Opgave

1. Læs dokumenterne i `test_dokumenter/`.
2. Udfyld de to markerede steder i `filter_template.py`:
   - `load_documents`: indlæs kun tilladte filtyper.
   - `filter_documents`: frasortér tydeligt dårlige dokumenter, og giv hvert
     frasorteret dokument en kort begrundelse.
3. Behold de dokumenter, der realistisk set ville forbedre retrieval-kvaliteten.
   Uændret beholder skabelonen alle 14 dokumenter.
4. Overvej, hvilke regler du bruger, for eksempel:
   - minimumslængde
   - forholdet mellem bogstaver og støj
   - typiske kladdeord som `draft`, `todo` eller `noter`
   - navigationstekst som `Home | About | Contact`
   - dubletter eller næsten identiske dokumenter

```powershell
python "øvelser/modul 2/filter_template.py"
```

Pointen med øvelsen er, at RAG-kvalitet starter før embeddings. Demoerne
`filter1.py`–`filter3.py` og `data_rensning_demo.py` i
[`demoer/modul 2`](../../demoer/modul%202/README.md) er gode til opvarmning.

## Opgave 2: Chunking

`chunking_template.py` chunker `demoer/modul 2/data/knowledge_base/chunking_workshop.md`.
Kursisterne skal udfylde to funktioner markeret med `TODO`:

- `chunk_fixed_size`: fixed-size chunking med tokens og overlap mellem nabo-chunks.
- `chunk_structure_aware`: del dokumentet ved Markdown-overskrifter, og del lange
  sektioner videre, så ingen chunk bliver for lang. Bevar gerne overskriften i chunken.

```powershell
python "øvelser/modul 2/chunking_template.py"
```

Før opgaverne er løst, kører skabelonen stadig, men melder "Ingen chunks endnu" og
viser 0 chunks for alle tre varianter.
[Chunking-laboratoriet](../../demoer/modul%202/chunking_lab/README.md) er godt til at
sammenligne resultatet med andre strategier bagefter.

### Forberedelse

Token-baseret chunking kræver tokenizer-data. Kør én gang med internetadgang, før
skabelonen og løsningen bruges:

```powershell
python tools/warmup_tokenizers.py
```

Bevar `.cache/tiktoken/` for at kunne køre opgaven offline bagefter.
