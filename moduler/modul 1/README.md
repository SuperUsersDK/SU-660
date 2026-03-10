# Avanceret Prompt Engineering Demo

Dette modul indeholder et samlet demo til undervisning i avanceret prompt engineering.

Demoen er lavet som en lille dokument-assistent, der svarer paa spoergsmaal om:

- chunking
- embeddings
- RAG
- overlap

Formaalet er at vise, hvordan en simpel LLM-applikation udvikler sig fra en naiv prompt til en mere robust loesning med bedre styring af kontekst, memory, tokens og outputformat.

## Laeringsmaal

Efter demoen skal deltagerne kunne forklare:

- forskellen paa statiske og dynamiske prompts
- hvorfor memory er nyttigt i samtaler
- hvorfor memory og kontekst koster tokens
- hvordan man budgetterer plads i prompten
- hvorfor kontrolleret output er vigtigt i software

## Filer

- `advanced_prompt_engineering_demo.py`: det samlede demo-script
- `data/doc_chunking.md`: lille tekst om chunking
- `data/doc_embeddings.md`: lille tekst om embeddings
- `data/doc_rag.md`: lille tekst om RAG
- `data/doc_overlap.md`: lille tekst om overlap

## Koer demoen

```bash
python "moduler/modul 1/advanced_prompt_engineering_demo.py"
```

Demoen bruger en lille mock-model, saa den kan koeres uden API-noegler eller eksterne services.

## Demoens struktur

### 1. Statisk prompt

Demoen starter med en helt simpel prompt:

- ingen kontekst
- ingen outputstyring
- ingen memory

Pointen er at vise, at en model godt kan give et svar, men at man har meget lidt kontrol.

### 2. Statisk prompt med regler

Her tilfoejes:

- rolle
- regler
- oensket stil
- begraensning paa svarlaengde

Pointen er, at selv en fast prompt kan give mere ensartede og nyttige svar.

### 3. Dynamisk prompt

Her indsættes runtime-data i prompten:

- spoergsmaal
- kontekst fra en lille vidensbase
- regler for svaret

Pointen er, at dynamiske prompts er noedvendige i rigtige applikationer som RAG, assistants og tool workflows.

### 4. Memory-styring

Her vises forskellen mellem:

- et opfoelgende spoergsmaal uden memory
- et opfoelgende spoergsmaal med samtalehistorik

Pointen er, at modellen bedre kan forstaa referencer som `de`, `det` og `den metode`, naar relevant historik er med.

### 5. Token-budgetering

Her vises:

- hvor promptens tokens bliver brugt
- hvordan memory og kontekst fylder mest
- hvordan man kan trimme historik og kontekst

Pointen er, at prompt engineering ogsaa handler om pladsstyring, ikke kun formulering.

### 6. Kontrol over output

Til sidst vises, hvordan man gaar fra frit tekstsvar til et mere stabilt JSON-format.

Pointen er, at kontrolleret output er lettere at:

- parse i kode
- validere
- sende videre i pipelines
- bruge i API'er

## Forslag til undervisningsflow

En god maade at bruge demoen paa er:

1. Koer scriptet fra start til slut uden afbrydelser.
2. Gaa derefter tilbage til hver del og diskuter, hvad der blev forbedret.
3. Sporg deltagerne, hvilket problem hver forbedring loeser.
4. Tal om tradeoffs, fx at mere memory giver bedre sammenhaeng, men ogsaa hoejere tokenforbrug.

## Gode spoergsmaal til rummet

- Hvad mangler den naive prompt?
- Hvad bliver bedre, naar vi tilfoejer faste regler?
- Hvorfor er dynamisk kontekst vigtig i RAG?
- Hvornar er fuld historik en daarlig ide?
- Hvad ville I skære vaek foerst, hvis prompten blev for lang?
- Hvorfor er JSON-output ofte bedre end fritekst i software?

## Udvidelser

Hvis du vil bygge videre paa demoen, kan du:

- erstatte mock-modellen med et rigtigt API-kald
- tilfoeje tiktoken eller tilsvarende til mere praecis tokenmaaling
- udvide vidensbasen med flere dokumenter
- tilfoeje retrieval-score eller citations
- lade deltagerne omskrive prompten og sammenligne resultater
