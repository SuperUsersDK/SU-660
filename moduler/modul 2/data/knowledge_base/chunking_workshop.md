# Chunking Workshop Notes

## Why chunking matters

Chunking decides what the retrieval system is even able to find.
If a document is split badly, the right information may exist in the source material but still never appear in the retrieved context.
That means the model may answer vaguely even when the knowledge base technically contains the answer.

In practice, chunking is often one of the most important design choices in a RAG system.
It affects retrieval precision, recall, latency, token cost, and how easy it is to explain why a chunk was returned.

## Fixed-size chunking

Fixed-size chunking is the simplest baseline.
You define a target size and split the text into windows of roughly the same length.
This approach is easy to implement and easy to reason about during early experiments.

The downside is that fixed-size chunking can cut directly through a sentence, a list, or a section boundary.
When that happens, one chunk may contain only half of the idea and the next chunk may contain the rest.
That is exactly why overlap is often introduced.

## Overlap and boundary protection

Overlap means that some content from the end of one chunk is repeated at the beginning of the next chunk.
This reduces the risk that important context disappears at a chunk boundary.
If a key sentence sits right between two chunks, overlap can improve recall.

Overlap is useful, but it is not free.
More overlap means more chunks, more redundancy, more storage, and more tokens during retrieval.
If overlap is too high, the top-k results may become repetitive instead of informative.

## Sentence-based chunking

Sentence-based chunking tries to preserve complete thoughts.
Instead of cutting after a fixed number of tokens, it groups whole sentences together.
This often makes chunks easier to read and easier to cite.

However, sentence-based chunking has its own tradeoffs.
Sentence lengths vary a lot, so chunk sizes may become uneven.
Some chunks can end up very short while others become surprisingly long.

## Structure-aware chunking

Markdown and documentation often contain headings, lists, and subsections.
Those structural markers can be used as natural chunk boundaries.
If a heading says "Overlap and boundary protection", it is usually helpful to keep that section together.

Structure-aware chunking is especially useful for manuals, runbooks, policies, and onboarding guides.
It improves explainability because retrieved chunks can be traced back to a clear section title.
It also tends to produce chunks that align better with how humans understand the document.

## Semantic chunking

Semantic chunking tries to group sentences that belong together conceptually.
The goal is not only to preserve syntax, but to preserve meaning.
In production systems, this may use embeddings or other similarity signals to decide where chunk boundaries should fall.

Semantic chunking can be powerful, but it is also more complex and more expensive.
It introduces more tuning parameters and often requires stronger observability to debug.
For teaching, a simplified approximation is often enough to show the idea.

## Example retrieval questions

- Why does overlap help around chunk boundaries?
- What is the downside of too much overlap?
- When is structure-aware chunking a better choice than fixed-size chunking?
- Why can sentence-based chunking be easier to explain to users?
- What extra complexity does semantic chunking introduce?

## Practical recommendation

Start with a simple baseline before reaching for advanced methods.
Fixed-size chunking without overlap is useful as a control condition.
Then test overlap, sentence grouping, and structure-aware chunking on the same document and compare the outputs.

When you compare strategies, do not only look at chunk count.
Also inspect whether the chunks are readable, whether they preserve complete ideas, and whether a retrieved chunk would make sense if shown directly to a user or developer.
