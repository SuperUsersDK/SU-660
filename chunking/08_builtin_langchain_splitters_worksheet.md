# Worksheet: Built-in Chunking (LangChain) - Demo 08

Dette ark bruges sammen med:

- `chunking/08_builtin_langchain_splitters.py`

Formaal:

- traene observation og tolkning af chunking-output
- skelne mellem top-1 score, top-k kvalitet og chunk-nytte
- oevе valg af strategi til forskellige dokumenttyper

## Del 1: Foer demoen (hypoteser)

Skriv dine forventninger foer demoen koeres:

1. Hvilken strategi tror du giver bedst retrieval paa almindelig plain text?
- [ ] CharacterTextSplitter
- [ ] RecursiveCharacterTextSplitter
- [ ] TokenTextSplitter
- [ ] Det afhænger (forklar)

Begrundelse:

..............................................................................

2. Hvilken strategi tror du er bedst til markdown policies/manuals?
- [ ] CharacterTextSplitter
- [ ] RecursiveCharacterTextSplitter
- [ ] TokenTextSplitter
- [ ] MarkdownHeaderTextSplitter

Begrundelse:

..............................................................................

## Del 2: Observation under demoen

Koer:

```bash
python3 chunking/08_builtin_langchain_splitters.py
```

Udfyld mens output vises.

### A. CharacterTextSplitter

1. Antal chunks:

..............................................................................

2. Hvad laegger du maerke til ved chunk-graenserne?

- [ ] Ser naturlige ud
- [ ] Bryder ofte "midt i noget"
- [ ] Bevarer god lokal kontekst
- [ ] Giver noget stoey

Eksempel/kommentar:

..............................................................................

3. Hvordan virker top-hits?

- [ ] Gode og fokuserede
- [ ] Delvist relevante
- [ ] Mere stoey end forventet

Kommentar:

..............................................................................

### B. RecursiveCharacterTextSplitter

1. Antal chunks:

..............................................................................

2. Sammenlignet med Character:

- [ ] Bedre graenser
- [ ] Samme kvalitet
- [ ] Vaerre graenser

Hvorfor?

..............................................................................

3. Hvordan virker top-hits?

- [ ] Mere fokuserede
- [ ] Mere komplette
- [ ] Mere redundante

Kommentar:

..............................................................................

### C. TokenTextSplitter

1. Antal chunks:

..............................................................................

2. Hvad er den vigtigste forskel ift. char-baserede splitters?

..............................................................................

3. Hvornår ville du foretrække token-baseret chunking?

- [ ] Tidlig prototype
- [ ] Naar cost/tokenbudget er vigtigt
- [ ] Naar dokumenter er korte
- [ ] Produktion generelt

Kommentar:

..............................................................................

### D. MarkdownHeaderTextSplitter

1. Antal chunks:

..............................................................................

2. Hvilken metadata bliver bevaret i output?

..............................................................................

3. Hvorfor er det nyttigt for retrieval/citations/debugging?

..............................................................................

4. Fandt top-hit den forventede sektion (`Termination Clause`)?
- [ ] Ja
- [ ] Nej

Kommentar:

..............................................................................

## Del 3: Tolkning (det vigtigste)

1. Giver hoejeste top-1 score altid den bedste chunking-strategi?
- [ ] Ja
- [ ] Nej

Hvorfor / hvorfor ikke?

..............................................................................

2. Hvad er vigtigere i praksis end top-1 score alene?

- [ ] top-k kvalitet / recall
- [ ] chunk-læsbarhed
- [ ] metadata/citationsmulighed
- [ ] cost/latency
- [ ] alle ovenstaaende

Kommentar:

..............................................................................

3. Hvis to strategier giver lignende scores, hvad kan afgøre valget?

..............................................................................

## Del 4: Designvalg (mini-case)

Vælg strategi + startparametre for hver case.

### Case 1: Policy / kontrakt (markdown eller struktureret tekst)

Valgt strategi:

..............................................................................

Chunk size (ca.):

..............................................................................

Overlap:

..............................................................................

Nødvendig metadata:

..............................................................................

### Case 2: Stor PDF-rapport (meget tekst, svag struktur)

Valgt strategi:

..............................................................................

Chunk size (ca.):

..............................................................................

Overlap:

..............................................................................

Ekstra hensyn (rensning, page metadata, etc.):

..............................................................................

### Case 3: FAQ / Knowledge Base artikler

Valgt strategi:

..............................................................................

Chunk size (ca.):

..............................................................................

Overlap:

..............................................................................

## Del 5: Opsamling (2 minutter)

Skriv dine 3 vigtigste takeaways:

1. ...........................................................................
2. ...........................................................................
3. ...........................................................................

## Bonus (for hold med ekstra tid)

Forslaa en lille evaluering, der kan sammenligne 2 chunking-strategier fair:

- Hvilke spørgsmål vil du bruge?
- Hvad vil du maale?
- Hvad holder du konstant?

Svar:

..............................................................................
..............................................................................
..............................................................................

