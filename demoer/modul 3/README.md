# Modul 3: Agents, tools og RAG

Modulet viser, hvordan en agent vælger tools og kombinerer svar fra dokumenter,
en lokal database og eksterne API'er. Brug eksemplerne i rækkefølgen nedenfor.
Alle kommandoer køres fra repository-roden med projektets `.venv` aktiveret.

## Før timen

Følg [fælles opsætning](../../README.md), og kontrollér:

- `OPENAI_API_KEY` er sat i `.env`, og kontoen har adgang til den valgte model.
- `OPENAI_MODEL` er som standard `gpt-6-luna`; embeddings bruger
  `OPENAI_EMBED_MODEL`, som er `text-embedding-3-small` som standard.
- `QDRANT_API_KEY` matcher Compose-konfigurationen. Python bruger normalt
  `QDRANT_URL=http://localhost:6333`; n8n-containeren bruger `http://qdrant:6333`.
- Der er internetadgang til OpenAI. Første agent bruger også Wikipedia og GitHub.
- Kør `python tools/warmup_tokenizers.py` med internetadgang før undervisningen.

Start de lokale services:

```powershell
docker compose up -d qdrant n8n
```

De to Python-agents indlæser selv `data/*.md` og klargør deres SQLite-eksempeldata.
Deres opstarts- og embedding-kald sker før de interaktive pauser.

## 1. Agent med flere datakilder

```powershell
python "demoer/modul 3/agent_tools_langchain_demo.py"
```

Tryk Enter ved pauserne. Det faste spørgsmål beder om fire typer oplysninger:

| Tool | Datakilde | Hvad holdet skal se |
|---|---|---|
| `search_documents` | Qdrant | Dokumenternes forklaring af metadata-filtre |
| `lookup_database` | SQLite | `vector-search` ejes af Platform Team og har SLA på 4 timer |
| `web_search` | Wikipedia summary API | Et opslag om LangChain; tool'et er et Wikipedia-opslag |
| `github_repo_stats` | GitHub REST API | Aktuelle stjerner, forks og åbne issues for `langchain-ai/langchain` |

Konsollen viser `TOOL docs`, `TOOL db`, `TOOL web` og `TOOL api`, når tools kaldes.
Agenten vælger rækkefølgen og kan kalde et tool flere gange. De konkrete webtal
kan ændre sig. Et eksternt opslag kan fejle; brug det til at diskutere, hvordan
agenten beskriver manglende evidens i sit svar.

Spørg holdet: Hvilke oplysninger kræver retrieval, hvilke kræver et databaseopslag,
og hvordan kan vi se, at svaret er baseret på tool-resultater?

## 2. Agent, der kombinerer evidens og beregning

```powershell
python "demoer/modul 3/advanced_langchain_agent_demo.py"
```

Scenariet er et `high`-severity outage på `vector-search`, som har varet 6,5 timer.
Se efter disse dele af tool-forløbet:

| Tool | Forventet bidrag |
|---|---|
| `lookup_service` | Platform Team, SLA på 4 timer og eksempel-kontakt til on-call |
| `search_documents` | Evidens om observability og metadata-filtre |
| `calculate` | `6.5 - 4 = 2.5` timer over SLA |
| `lookup_policy` | Eskalér straks til on-call og opdatér interessenter hvert 30. minut |

Svaret bruger sektionerne Oversigt, Evidens, Risici og Næste handlinger.
Tool-rækkefølge og ordlyd kan variere. Serviceoplysninger og kontakter er syntetiske;
demoen viser en anbefaling og sender ingen beskeder.

Spørg holdet: Hvorfor er dette en agent frem for en fast kæde? Hvilke fakta
kommer fra dokumenter, database, beregning og policy?

## 3. Klassisk RAG som n8n-workflow

Åbn `http://localhost:5678`, opret den lokale brugerkonto, og importér
`demoer/modul 3/Klassisk RAG Pipeline.json`. Vælg egne OpenAI- og Qdrant-credentials
i de importerede nodes; credential-ID'er fra eksporten skal tilknyttes lokalt.

1. Sæt Qdrant-credentialens URL til `http://qdrant:6333` og dens nøgle til Compose-nøglen.
2. Kør Manual Trigger-grenen: læs filer → split dokumenter → embeddings → Qdrant.
3. Åbn Chat Trigger, og spørg eksempelvis: "Hvad gør metadata-filtre i RAG?"
4. Følg retrieval med `topK=3`, prompten, modellen og svaret.

Diskutér forskellen på denne faste RAG-pipeline og de to agents' valg af tools.
Workflowet skal gennemprøves lokalt med credentials før timen.

## 4. Billedlighed med CLIP (foto-webapp)

Demoen embedder billeder lokalt med CLIP, gemmer dem i Qdrant og viser dem i en webapp.
Den kræver ingen OpenAI-nøgle. Detaljer står i [foto-demoets vejledning](photos/README.md).

**1. Hent billeder.** Billedmappen `photos/data/` er ikke i git, så billederne skal
hentes på hver maskine. Scriptet henter 20 billeder i hver af 10 tydelige kategorier
(bil, fly, skib, fugl, insekt, blomst, svamp, kirke, mad og bjerg) fra Wikimedia
Commons. Det kræver internet og tager ca. 10 minutter; en afbrudt kørsel fortsætter,
når scriptet køres igen. Egne `.jpg`-billeder kan i stedet lægges i `photos/data/`.

```powershell
python "demoer/modul 3/photos/fetch_sample_photos.py"
```

**2. Indlæs billederne i Qdrant** (embedder og tagger dem):

```powershell
python "demoer/modul 3/photos/ingest_photos.py"
```

Siden **Sammenlign modeller** kræver desuden den store CLIP-model (ca. 1,7 GB):

```powershell
python "demoer/modul 3/photos/ingest_photos.py" --model openai/clip-vit-large-patch14
```

**3. Start webappen** og åbn `http://127.0.0.1:8010`:

```powershell
cd "demoer/modul 3/photos"
uvicorn app:app --port 8010
```

Webappen har siderne **Søg** (billede eller tekst, filtreret på tag), **Kort** (alle
billeder i 2D eller 3D, t-SNE eller PCA), **Hvad ser modellen?**, **Dubletter** og
**Sammenlign modeller**. Kør trin 1 og 2 før timen, så billeder og modelvægte er
downloadet; derefter virker demoen uden internet.

## 5. Foto-agent: CLIP-søgning + vision-model

Kræver, at foto-ingest er kørt (afsnit 4).

```powershell
python "demoer/modul 3/photo_agent_demo.py"
```

Med eksempelbillederne (`PHOTO_LABEL_SET=eksempler`) skal agenten finde de tre
bedste billeder af fugle og af insekter, beskrive dem og anbefale et forsidebillede
til en plakat om dyreliv. Med `PHOTO_LABEL_SET=dykning` spørger den i stedet efter
vragbilleder med en dykker.

| Tool | Datakilde | Hvad holdet skal se |
|---|---|---|
| `list_photo_tags` | Qdrant payload | Hvilke zero-shot tags arkivet har |
| `search_photos` | CLIP-tekstvektor + Qdrant | Agenten skriver selv en engelsk søgetekst og kan filtrere på et tag |
| `describe_photo` | Vision-model (`OPENAI_MODEL`) | Modellen ser selve billedet og bekræfter eller afviser kandidaten |

Konsollen viser `TOOL search` og `TOOL vision`. Agenten kalder ofte flere tools
parallelt, fx én søgning efter fugle og én efter insekter, og beskriver derefter
kandidaterne. Spørg holdet: Hvorfor er CLIP god til at finde kandidater, men en
vision-model nødvendig for at bekræfte dem? Hvad koster de mange vision-kald?

## Hvad nulstilles ved genkørsel?

| Demo | Adfærd |
|---|---|
| Agent med flere kilder | Genskaber `modul3_agent_tools_demo` og nulstiller `service_info` i `data/agent_demo.db` |
| Avanceret agent | Genskaber `modul3_advanced_agent_demo` og nulstiller `services` og `escalation_policy` i `data/advanced_agent_demo.db` |
| Foto-ingest | Genskaber modellens collection (standard `modul3_photos_demo`) fra billederne i `photos/data/` og tagger dem |
| Foto-tagging | Overskriver kun `tag` og `tag_score` i payload |
| n8n | Bruger `modul3_demo`; ingest-grenen indsætter data og nulstiller ikke automatisk collectionen |

Nulstillingerne giver Python-demoerne et fast udgangspunkt. Brug de nævnte
collections til undervisningsdata. Undgå at genkøre n8n-ingest unødigt, da det
kan tilføje dubletter. `docker compose down` stopper services og bevarer volumes.

## Hurtig fejlsøgning

- Manglende OpenAI-nøgle: kontrollér `.env` i repository-roden.
- Qdrant 401: kontrollér, at nøgle og URL passer til Python eller n8n som beskrevet ovenfor.
- Forbindelsesfejl: kontrollér `docker compose ps` og internetadgang til de eksterne API'er.
- Tokenizer-download: kør warmup med netadgang, og bevar `.cache/tiktoken/`.
- Foto-demo uden resultater: kontrollér billedmappen, og kør `ingest_photos.py`.
- `ModuleNotFoundError` (fx `dotenv`): `.venv` er ikke aktiveret, så systemets Python bruges.

## Filer

| Fil | Indhold |
|---|---|
| `agent_tools_langchain_demo.py` | Agent med flere datakilder (afsnit 1) |
| `advanced_langchain_agent_demo.py` | Agent med evidens og beregning (afsnit 2) |
| `Klassisk RAG Pipeline.json` | n8n-workflow (afsnit 3) |
| `photos/` | Foto-webapp og scripts (afsnit 4) |
| `photo_agent_demo.py` | Foto-agent (afsnit 5) |
| `data/*.md` | Dokumenter, som agenterne og n8n-workflowet indekserer |
| `data/*.db` | SQLite-eksempeldata til agenterne; nulstilles ved hver kørsel |
| `frankenstein.txt` | Romanen fra Project Gutenberg. Bruges ikke af demoerne, men kan kopieres til `data/` for at vise n8n-ingest af en lang tekst |
