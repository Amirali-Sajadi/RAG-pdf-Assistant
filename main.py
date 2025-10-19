import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from groq import Groq
from dotenv import load_dotenv
import uuid
import os
import traceback
from pdf_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from output_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAQQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer
)

def load_chunks_from_context(ctx):
    """Standalone version of _load() for testing and Inngest usage."""
    pdf_path = ctx.event.data.get("pdf_path")
    source_id = ctx.event.data.get("source_id", pdf_path)
    chunks = load_and_chunk_pdf(pdf_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        result = RAGChunkAndSrc(chunks=chunks, source_id=source_id)
        return result

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))


    try:
        print("🟦 Step 1: Loading and chunking PDF...")
        chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx).model_dump())
        chunks_and_src = RAGChunkAndSrc(**chunks_and_src)
        print(f"✅ Loaded {len(chunks_and_src.chunks)} chunks from {chunks_and_src.source_id}")

        print("🟦 Step 2: Embedding and upserting...")
        ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src).model_dump())
        ingested = RAGUpsertResult(**ingested)
        print(f"✅ Embedded and upserted {ingested.ingested} chunks.")
        return ingested.model_dump()

    except Exception as e:
        print("❌ ERROR in RAG pipeline:", e)
        print(traceback.format_exc())
        raise

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
)

async def rag_query_pdf(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k).model_dump())
    found = RAGSearchResult(**found)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:{question}\n"
        "Answer concisely using the given context."
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system",
             "content": "You are a helpful AI assistant that answers questions using only the provided context."},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf, rag_query_pdf])
