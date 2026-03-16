# Instructor Notes / Facit: Built-in Chunking (LangChain) - Demo 08

Bruges sammen med:

- `chunking/08_builtin_langchain_splitters.py`
- `chunking/08_builtin_langchain_splitters_worksheet.md`

Formaal:

- give underviseren et hurtigt facit og talepunkter
- fremhaeve typiske misforstaaelser
- støtte diskussion frem for "én rigtig score"

## Vigtig framing (sig dette tidligt)

Demoen bruger en **didaktisk lexical score** (ord-overlap / cosine-lignende), ikke embeddings.

Det betyder:

- scores er gode til sammenligning i demoen
- scores er ikke produktionssandhed
- top-1 score alene er ikke nok til at vurdere chunking-kvalitet

## Del 1: Hypoteser (forventede svar)

### Spm 1: Bedst retrieval paa plain text?

Ofte forventet svar:

- `RecursiveCharacterTextSplitter` eller `TokenTextSplitter`

Hvorfor:

- recursive bevarer ofte bedre graenser end ren character split
- token split matcher model-budget bedre (især i produktion)

God diskussion:

- "bedst" afhænger af query-type og dokumenttype
- top-k kvalitet og chunk-læsbarhed er vigtigere end kun top-1 score

### Spm 2: Bedst til markdown policies/manuals?

Forventet svar:

- `MarkdownHeaderTextSplitter`

Hvorfor:

- bevarer struktur som metadata (`h1`, `h2`, ...)
- lettere at forklare, citere og debugge
- ofte bedre alignment med hvordan brugere spørger ("what does section X say?")

## Del 2: Observationer pr. splitter (facit-guidance)

### A. CharacterTextSplitter

Forventede observationer:

- chunks kan bryde "midt i noget" (semantisk kluntede graenser)
- retrieval kan stadig fungere rimeligt
- top-hits kan indeholde relevant info, men chunks er ofte mindre naturlige

Hvad kursister ofte siger (korrekt):

- "Den virker, men er lidt grim"

Typisk misforstaaelse:

- "Hvis den virker her, er den altid fin"

Korriger med:

- vis at den kan være sårbar på struktur-tunge dokumenter, tabeller og citationskrav

### B. RecursiveCharacterTextSplitter

Forventede observationer:

- mere naturlige chunkgraenser end Character
- ofte stærk retrieval-kvalitet i top-k
- godt kompromis mellem enkelhed og kvalitet

Underviser-pointe:

- dette er ofte bedste generelle default at starte med

Typisk misforstaaelse:

- "Recursive er altid bedst"

Korriger med:

- heading-aware kan være bedre på strukturerede docs
- token-aware kan være bedre når budget/cost er vigtigt

### C. TokenTextSplitter

Forventede observationer:

- chunk count og graenser afviger fra char-baserede splitters
- mere relevant ift. reelle LLM-kontekstbudgetter
- output kan se lidt mindre "menneskeligt" ud end heading-aware splitting

Underviser-pointe:

- token chunking handler ofte mere om drift/cost/kontekststyring end "pæneste chunks"

Typisk misforstaaelse:

- "Token splitter er automatisk bedre retrieval"

Korriger med:

- det er bedre budgetkontrol, ikke nødvendigvis bedre retrieval alene

### D. MarkdownHeaderTextSplitter

Forventede observationer:

- få, semantisk meningsfulde chunks
- tydelig metadata (`h1`, `h2`)
- termination-query rammer typisk `Termination Clause` chunket

Underviser-pointe:

- strukturmetadata er enormt værdifuldt til citations, filtrering og debugging

Typisk misforstaaelse:

- "Heading-aware virker på alle PDF’er"

Korriger med:

- kun hvis struktur kan bevares / udledes pålideligt

## Del 3: Tolkning (facit)

### "Giver højeste top-1 score altid bedste strategi?"

Facit:

- Nej

Begrundelse (det vigtigste):

- top-1 score siger ikke alt om top-k kvalitet
- chunk kan have høj score men mangle kritisk kontekst
- retrieval skal vurderes sammen med:
  - chunk-læsbarhed
  - grounded answer quality
  - metadata/citationsmulighed
  - redundans

### "Hvad er vigtigere end top-1 score alene?"

Facit:

- Ofte "alle ovenstående" i worksheet’et:
  - top-k kvalitet / recall
  - chunk-læsbarhed
  - metadata/citationsmulighed
  - cost/latency

## Del 4: Mini-case (forventede svar-retninger)

### Case 1: Policy / kontrakt (markdown eller struktureret tekst)

Typisk godt svar:

- Strategi: `MarkdownHeaderTextSplitter` + evt. recursive inde i sektioner (hybrid)
- Chunk size: moderate chunks (fx 400-900 tokens hvis videre split)
- Overlap: lav-moderat (5-15%)
- Metadata: `section`, `page` (hvis PDF), `source_file`, `chunk_id`

Hvad du leder efter:

- kursisten argumenterer ud fra citationsbehov og struktur

### Case 2: Stor PDF-rapport (svag struktur)

Typisk godt svar:

- Strategi: `RecursiveCharacterTextSplitter` (evt. token-aware variant)
- Chunk size: fx 600-1000 tokens
- Overlap: 10-20%
- Ekstra: page metadata, cleanup af headers/footers, evaluer top-k

Hvad du leder efter:

- forståelse for noisy PDFs og behov for metadata/rensning

### Case 3: FAQ / KB artikler

Typisk godt svar:

- Strategi: sentence/paragraph-ish (eller recursive med mindre chunks)
- Chunk size: kortere (fx 150-500 tokens)
- Overlap: lav-moderat

Hvad du leder efter:

- match mellem korte spørgsmål/svar og mindre chunks

## Typiske misforstaaelser (opsummeret)

1. "Højeste score = bedste chunking"
- Nej, se top-k + svarkvalitet + metadata

2. "Token chunking er altid bedst"
- Nej, det er bedst til budgetstyring og ofte produktion, men ikke automatisk bedst retrieval

3. "Recursive er altid bedst"
- Nej, heading-aware/hybrid kan være bedre på strukturerede dokumenter

4. "Built-ins er kun til prototyper"
- Nej, built-ins bruges ofte i produktion som baseline eller permanent løsning

5. "Hvis chunks ser pæne ud, er retrieval godt"
- Ikke nødvendigvis - mål retrieval/QA på rigtige spørgsmål

## Forslag til opsamling (2-3 min)

Spørg holdet:

- Hvilken built-in ville I vælge som default i jeres team - og hvorfor?
- Hvad ville få jer til at skifte strategi?
- Hvilke metrics ville I måle i uge 1?

## Bonus: Hurtig evalueringsopgave (hjemmeopgave)

Bed kursisterne lave en mini-sammenligning med 2 strategier på egne dokumenter:

- samme docs
- samme spørgsmål
- samme top-k
- log retrieval hits + scores
- vurder groundedness/citations

Leverance (kort):

- valgt strategi
- begrundelse
- 3 observationer
- 1 næste eksperiment

