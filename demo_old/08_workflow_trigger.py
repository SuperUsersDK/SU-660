import json
from datetime import datetime, timezone

from common import SAMPLE_DOCS, append_jsonl, build_chunks_sentence


def build_ingest_event(doc: dict[str, str]) -> dict:
    return {
        "event_type": "document.ingested",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc["id"],
        "title": doc["title"],
        "source": doc["source"],
        "category": doc["category"],
    }


def build_eval_trigger(doc_ids: list[str]) -> dict:
    return {
        "event_type": "evaluation.trigger",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "new_content_batch",
        "doc_ids": doc_ids,
        "eval_suite": "rag_smoke_tests_v1",
    }


def main() -> None:
    print("=== Demo: Simpelt workflow for dokumentbehandling eller evalueringstrigger ===")

    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)
    print(f"Antal genererede chunks i ingest-step: {len(chunks)}")

    ingest_events = [build_ingest_event(doc) for doc in SAMPLE_DOCS[:2]]
    eval_trigger = build_eval_trigger([doc["id"] for doc in SAMPLE_DOCS[:2]])

    for event in ingest_events + [eval_trigger]:
        append_jsonl("demo/output/workflow_events.jsonl", event)

    print("\nEksempel pa payloads (kan sendes til fx n8n webhook):")
    print(json.dumps(ingest_events[0], indent=2, ensure_ascii=True))
    print(json.dumps(eval_trigger, indent=2, ensure_ascii=True))
    print("\nSkrevet til: demo/output/workflow_events.jsonl")


if __name__ == "__main__":
    main()

