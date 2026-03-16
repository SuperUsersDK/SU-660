## Opgave: Filtrer brugbare dokumenter til RAG

I denne oevlelse skal deltagerne arbejde med et lille dokumentsaet med blandet kvalitet.

Maallet er at skrive et filter, der kun bevarer de dokumenter, som er gode nok til at blive brugt i et RAG-system.

Dokumentsaettet indeholder med vilje en blanding af:

- rigtige faglige tekster
- navigationstekst
- kladder
- dubletter eller naesten-dubletter
- korte irrelevante tekster

### Opgave

1. Laes dokumenterne i mappen `test_dokumenter`.
2. Skriv et filter, der frasorterer tydeligt daarlige dokumenter.
3. Behold de dokumenter, der realistisk set ville forbedre retrieval-kvaliteten.
4. Overvej hvilke regler du bruger, for eksempel:
   - minimumslaengde
   - forhold mellem bogstaver og stoej
   - typiske kladdeord som `draft`, `todo` eller `noter`
   - navigationstekst som `Home | About | Contact`
   - dubletter eller naesten identiske dokumenter

Pointen med oevlelsen er, at RAG-kvalitet starter foer embeddings.
