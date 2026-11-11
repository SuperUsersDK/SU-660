# RAG demo

From the repository root:

```powershell
python "demoer/modul 2/big_demo/ingest.py"
python "demoer/modul 2/big_demo/chat_cli.py"
```

The CLI first displays a diagnostic search using your original question. The
agent then chooses its own tool calls and may rewrite the retrieval query, so
those diagnostic hits can differ from the context used in its answer.

Ingestion synchronizes the selected document directory with Qdrant. After every
file has loaded and all new chunks have been stored successfully, it removes old
chunks for edited, emptied, or deleted files. An existing empty directory removes
its previously indexed chunks; a missing directory raises an error.

Use `--document-dir PATH` to select another directory. Paths relative to the demo
directory are supported. Each directory has its own chunk IDs and ownership tag,
so synchronizing one directory preserves documents indexed from other directories.
Run one ingestion at a time for a given directory and collection.

`--force` re-embeds current chunks and still removes obsolete chunks. The first
run after this update replaces old IDs with IDs that include the full source
path. Legacy chunks are removed only when their stored source path identifies
them as belonging to the selected directory.

Run the offline regression suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```
