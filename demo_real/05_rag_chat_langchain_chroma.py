from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from demo_real.common_real import chroma_dir, get_models, load_env, sample_documents


def build_vectorstore() -> Chroma:
    _, embed_model = get_models()
    embeddings = OpenAIEmbeddings(model=embed_model)
    docs = [
        Document(
            page_content=item["text"],
            metadata={"id": item["id"], "source": item["source"], "category": item["category"]},
        )
        for item in sample_documents()
    ]
    vs = Chroma(
        collection_name="su660_demo",
        embedding_function=embeddings,
        persist_directory=chroma_dir(),
    )
    vs.reset_collection()
    vs.add_documents(docs)
    return vs


def main() -> None:
    load_env()
    chat_model, _ = get_models()
    llm = ChatOpenAI(model=chat_model, temperature=0)
    vs = build_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    question = "Hvorfor er metadata-filters nyttige i RAG, og hvad skal vi logge?"
    hits = retriever.invoke(question)

    context = "\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}"
        for d in hits
    )
    prompt = ChatPromptTemplate.from_template(
        "Svar kun ud fra konteksten. Hvis noget mangler, sig det tydeligt.\n\nKontekst:\n{context}\n\nSporgsmal:\n{question}"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    print("Top hits:")
    for d in hits:
        print(f"- {d.metadata.get('source')}: {d.page_content}")
    print("\nSvar:")
    print(response.content)


if __name__ == "__main__":
    main()

