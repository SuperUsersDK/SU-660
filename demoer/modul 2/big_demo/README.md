# RAG-demo med agent og Qdrant

Demoen indekserer en mappe med blandede filer (Markdown, PDF og kode) i Qdrant og lader
en agent besvare spørgsmål om dem i terminalen. Agenten vælger selv, hvornår den søger,
og kan omformulere søgningen.

## Forudsætninger

- `.venv` er aktiveret (se [fælles opsætning](../../../README.md)).
- `OPENAI_API_KEY` er sat i `.env`. Embeddings laves med `EMBEDDING_MODEL`
  (standard `text-embedding-3-small`), og svarene med `OPENAI_MODEL`.
- Qdrant kører (`docker compose up -d qdrant`), og `QDRANT_API_KEY` passer til
  `docker-compose.yml`. Chunks gemmes i `QDRANT_COLLECTION` (standard `rag_data`).

## Kørsel

Fra repository-roden:

```powershell
python "demoer/modul 2/big_demo/ingest.py"
python "demoer/modul 2/big_demo/chat_cli.py"
```

Skriv spørgsmål ved `>`, og afslut med `exit`. Eksempler:

- "Hvad handler dokumenterne om?"
- "Hvordan genstarter man et deployment i Kubernetes?"

CLI'en viser først en diagnostisk søgning med det oprindelige spørgsmål (score og
kilde for de 8 bedste chunks). Agenten vælger derefter selv sine tool-kald og kan
omskrive søgningen, så dens kontekst kan afvige fra de diagnostiske hits. Svaret
angiver kilderne som `[source=...]`.

## Agentens tools

| Tool | Gør |
|---|---|
| `retrieve` | Søger i Qdrant efter de mest relevante chunks |
| `calculator` | Regner et simpelt regnestykke ud, fx `23*47` |
| `doc_stats` | Tæller chunks pr. kildefil i de fundne resultater |
| `search_in_context` | Finder en sætning i den hentede kontekst og viser teksten omkring den |

## Ingest

`ingest.py` synkroniserer dokumentmappen (standard `data/`) med Qdrant og læser
`.pdf`, `.md`, `.py` og `.cs`. Når alle filer er indlæst og alle nye chunks er gemt,
fjernes gamle chunks for filer, der er ændret, tømt eller slettet. En tom mappe fjerner
mappens tidligere chunks; en mappe, der ikke findes, giver en fejl.

- `--document-dir STI` vælger en anden mappe (relative stier regnes fra demo-mappen).
  Hver mappe har sine egne chunk-id'er, så synkronisering af én mappe bevarer
  dokumenter fra andre mapper.
- `--force` embedder alle chunks igen og fjerner stadig forældede chunks.
- Kør kun én ingest ad gangen for samme mappe og collection.

## Filer

| Fil | Indhold |
|---|---|
| `ingest.py` | Indlæser, chunker, embedder og synkroniserer filer med Qdrant |
| `chat_cli.py` | Terminal-chat med agenten |
| `rag_chain.py` | Vektor-store, retrieval og agent |
| `tools.py` | Agentens hjælpe-tools |
| `settings.py` | Indstillinger fra `.env` |
| `data/` | Kildedokumenterne, der indekseres (RAG-data, ikke eksempler, der skal køres) |

De offline regressionstests køres fra repository-roden:

```powershell
python -m unittest discover -s tests -v
```
