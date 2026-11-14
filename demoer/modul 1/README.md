# Modul 1: Prompt engineering

Modulet har fire demoer om prompts, hallucinationer og token-budgetter. Alle kører i
terminalen og holder pauser (tryk Enter), så underviseren kan diskutere outputtet.

| Demo | Viser | Kalder OpenAI |
|---|---|---|
| `advanced_prompt_engineering_demo.py` | Fra naiv prompt til robust løsning: regler, dynamisk kontekst, memory, token-budget og JSON-output | Ja |
| `hallucination.py` | Samme spørgsmål om en opdigtet artikel med en naiv og en forsigtig prompt | Ja |
| `tokenbudget.py` | Hvor tokens bruges i en prompt, og hvad trimning sparer | Nej |
| `tokenbudget2.py` | Komprimering af samtalehistorik med et rigtigt modelkald | Ja |

## Før timen

Følg [opsætningen i repository-roden](../../README.md), aktivér `.venv`, og sæt
`OPENAI_API_KEY` i `.env`. Demoerne bruger rigtige OpenAI API-kald og kræver
internetadgang. `OPENAI_MODEL` vælger modellen; standarden er `gpt-6-luna`.

Kør fra repository-roden:

```bash
python "demoer/modul 1/advanced_prompt_engineering_demo.py"
python "demoer/modul 1/hallucination.py"
python "demoer/modul 1/tokenbudget.py"
python "demoer/modul 1/tokenbudget2.py"
```

## Avanceret prompt engineering

`advanced_prompt_engineering_demo.py` er en lille dokument-assistent, der svarer på
spørgsmål om chunking, embeddings, RAG og overlap. Den viser, hvordan en simpel
LLM-applikation udvikler sig fra en naiv prompt til en mere robust løsning med bedre
styring af kontekst, memory, tokens og outputformat. Der bruges lokal, ordbaseret
retrieval over de fire tekster i `data/`.

### Læringsmål

Efter demoen skal deltagerne kunne forklare:

- forskellen på statiske og dynamiske prompts
- hvorfor memory er nyttigt i samtaler
- hvorfor memory og kontekst koster tokens
- hvordan man budgetterer plads i prompten
- hvorfor kontrolleret output er vigtigt i software

### Demoens dele

1. **Statisk prompt:** ingen kontekst, ingen outputstyring og ingen memory. Modellen
   kan godt svare, men man har meget lidt kontrol.
2. **Statisk prompt med regler:** rolle, regler, ønsket stil og begrænset svarlængde.
   Selv en fast prompt kan give mere ensartede og nyttige svar.
3. **Dynamisk prompt:** spørgsmål, kontekst fra en lille vidensbase og regler for
   svaret indsættes ved kørsel. Det er nødvendigt i RAG, assistenter og tool-workflows.
4. **Memory:** et opfølgende spørgsmål med og uden samtalehistorik. Med historik
   forstår modellen referencer som "de", "det" og "den metode".
5. **Token-budget:** hvor promptens tokens bruges, at memory og kontekst fylder mest,
   og hvordan historik og kontekst kan trimmes. Prompt engineering handler også om
   pladsstyring.
6. **Kontrol over output:** fra frit tekstsvar til JSON, som er lettere at parse,
   validere, sende videre i pipelines og bruge i API'er.

Tallene i token-delen er undervisningsestimater baseret på ordtælling. Instruktioner
om svarformat er en del af inputtet; plads til modellens svar skal reserveres separat.
Memory-resuméet er forberedt til eksempel-historikken.

## Hallucination

`hallucination.py` spørger om hovedresultaterne i en artikel, der ikke findes
("Quantum Governance in Nordic Municipalities" fra 2019):

1. **Naiv prompt:** "svar konkret og giv gerne detaljer". Svarer modellen selvsikkert
   på noget, der måske ikke findes?
2. **Forsigtig prompt:** modellen bliver bedt om at sige tydeligt, når den er usikker,
   og ikke opfinde detaljer.

Diskutér, hvordan prompten påvirker risikoen for opdigtede svar, og hvorfor RAG med
kilder er den egentlige løsning.

## Token-budget

- `tokenbudget.py` er en lokal regneøvelse med et kunstigt kontekstvindue på 800
  enheder og en særskilt svarreserve. Den viser, hvor meget hver del af prompten
  fylder, og hvad trimning sparer. Det er ikke modellens faktiske kontekstgrænse eller
  en måling med dens tokenizer.
- `tokenbudget2.py` lader modellen komprimere samtalehistorikken, sammenligner
  token-estimaterne før og efter og kører derefter den nye, kortere prompt.

## Forslag til undervisningsflow

1. Kør `advanced_prompt_engineering_demo.py` fra start til slut, og tryk Enter ved pauserne.
2. Gå derefter tilbage til hver del, og diskutér, hvad der blev forbedret.
3. Spørg deltagerne, hvilket problem hver forbedring løser.
4. Tal om tradeoffs, fx at mere memory giver bedre sammenhæng, men også højere tokenforbrug.
5. Vis `hallucination.py` som en advarsel, og `tokenbudget.py`/`tokenbudget2.py` som
   uddybning af token-delen.

## Gode spørgsmål til holdet

- Hvad mangler den naive prompt?
- Hvad bliver bedre, når vi tilføjer faste regler?
- Hvorfor er dynamisk kontekst vigtig i RAG?
- Hvornår er fuld historik en dårlig idé?
- Hvad ville I skære væk først, hvis prompten blev for lang?
- Hvorfor er JSON-output ofte bedre end fritekst i software?

## Udvidelser

- tilføj tiktoken eller tilsvarende til mere præcis tokenmåling
- udvid vidensbasen med flere dokumenter
- tilføj retrieval-score eller kildehenvisninger
- lad deltagerne omskrive prompten og sammenligne resultater

## Filer

| Fil | Indhold |
|---|---|
| `advanced_prompt_engineering_demo.py` | Den samlede prompt-engineering-demo |
| `hallucination.py` | Naiv og forsigtig prompt om en opdigtet artikel |
| `tokenbudget.py` | Lokal token-budget-regneøvelse |
| `tokenbudget2.py` | Memory-komprimering med et modelkald |
| `data/doc_chunking.md`, `doc_embeddings.md`, `doc_rag.md`, `doc_overlap.md` | Små tekster, som den avancerede demo henter kontekst fra |
