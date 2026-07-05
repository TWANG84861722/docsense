import logging

from query import retrieve_rounds
import config
from config import MAX_HISTORY_TURNS, MAX_TOKENS
import model_client

EARLY_STOP_MISSES = 5   # How many consecutive chunks the LLM judges NONE before stopping this
                        # round's map (currently the only stopping mechanism).

MAP_PROMPT_TMPL = """Does the following text excerpt directly answer or explicitly address the question?

Question: {question}

[{paper}  p.{page}  {section}]
{text}

Rules:
- If the excerpt explicitly contains facts that answer the question, extract only those facts concisely.
- If the excerpt does not directly address the question, respond with exactly: NONE
- Do not infer, generalize, or include loosely related content.
- NONE means zero direct relevance, not low relevance."""

REDUCE_PROMPT_TMPL = """The following were extracted from multiple scientific papers in response to:
"{question}"

Synthesize into a single comprehensive answer. Deduplicate and organize clearly.

Extracted findings:
{extractions}
"""

CONDENSE_PROMPT_TMPL = """Rewrite the LAST user question into a standalone, self-contained question \
**in English**, for searching an English scientific-paper corpus.
- Translate to English if the question is in another language.
- Resolve any references (it, the second one, that, these, ...) using the conversation history, \
replacing them with the specific names they refer to.
- If a gene symbol or technical term looks garbled (e.g. from speech recognition), correct it to the \
most likely intended term given the context.
- If it is already a standalone English question, return it unchanged.
Output ONLY the rewritten English question — no explanation, no quotes.

Conversation history:
{history}

Last user question: {question}

Rewritten standalone English question:"""

logger = logging.getLogger(__name__)

SYSTEM = """Use the retrieved evidence first.

When answering, structure your response as:

FACTS:
- Only statements directly supported by the provided context.

INFERENCE:
- Reasoned conclusions based on evidence.
- Clearly distinguish from facts.

CONFIDENCE:
- High / Medium / Low

If evidence is insufficient, explicitly say so.
Do not fabricate facts.
"""

def map_phase(question, chunks):
    """Run map step. Returns (extractions, sources, examined).

    examined = how many chunks this round actually looked at (on early stop = where it stopped;
    otherwise = the full batch size). The caller uses it to judge "did we page past the first
    CANDIDATE_K without stopping" and thus whether to deepen by another round.
    """
    extractions = []
    sources = []
    consecutive_misses = 0

    for i, chunk in enumerate(chunks):
        prompt_text = MAP_PROMPT_TMPL.format(
            question=question,
            paper=chunk["paper"],
            page=chunk["page"],
            section=chunk.get("section", ""),
            text=chunk["text"],
        )
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        # No longer hard-cutting by rerank score: every chunk is handed to the LLM to judge
        # "relevant or not". In a homogeneous corpus a relevant chunk's score can be quite low,
        # so a hard cut would kill it off before the LLM ever sees it.
        result = model_client.chat(messages, max_tokens=150).strip()

        if result and result != "NONE":
            extractions.append(result)
            sources.append(chunk)
            consecutive_misses = 0
            print(f"    chunk {i+1}/{len(chunks)}  found   (rerank={chunk.get('rerank_score', 0):.3f})")
        else:
            consecutive_misses += 1
            print(f"    chunk {i+1}/{len(chunks)}  skip [LLM:NONE]  (rerank={chunk.get('rerank_score', 0):.3f})  misses={consecutive_misses}")
            if consecutive_misses >= EARLY_STOP_MISSES:
                print(f"    Early stop.")
                return extractions, sources, i + 1      # examined = stopped here, looked at i+1

    return extractions, sources, len(chunks)            # went through the whole batch without stopping


def map_reduce(question):
    all_extractions = []
    all_sources = []

    for round_idx, chunks in enumerate(retrieve_rounds(question), 1):
        scores = [c["rerank_score"] for c in chunks]
        print(f"\n[Round {round_idx} — {len(chunks)} candidates (FAISS+BM25 union → reranked)]"
              f"  rerank: max={max(scores):.3f}  median={sorted(scores)[len(scores)//2]:.3f}  min={min(scores):.3f}")
        extractions, sources, examined = map_phase(question, chunks)
        all_extractions.extend(extractions)
        all_sources.extend(sources)

        # Keep-going rule: if this round "paged past the first CANDIDATE_K without stopping"
        # (examined > K) → lots of hits, deepen by another round; if it stopped early
        # (examined ≤ K) → we're done.
        if examined <= config.CANDIDATE_K:
            break
        print(f"[Round {round_idx} paged past {config.CANDIDATE_K} still productive (examined={examined}) → fetching next round]")

    if not all_extractions:
        return "No relevant information found.", []

    all_findings = "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(all_extractions))
    reduce_prompt_text = REDUCE_PROMPT_TMPL.format(
        question=question, extractions=all_findings
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": reduce_prompt_text},
    ]
    print(f"\n[Reducing {len(all_extractions)} findings...]")
    return model_client.chat(messages, max_tokens=MAX_TOKENS), all_sources


def condense_question(question, history):
    """Rewrite the question (in any language) into a standalone *English* question -- for retrieval.

    One LLM call does three things at once:
    1. Translate to English (the corpus is English; BM25 only matches literally, so it matches
       only within the same language, otherwise half the retrieval is crippled);
    2. Resolve references using the conversation history ("it / the second one" → the specific
       name; the referent is usually in the previous answer);
    3. Fix speech-recognition errors on gene names / technical terms along the way (guessing
       back from context).
    If it is already English *and* there is no history (no references to resolve), return it as-is
    without calling the LLM (saving one call).
    """
    if question.isascii() and not history:      # already English and no history to resolve → as-is, save an LLM call
        return question
    hist = "(none)"
    if history:
        lines = []
        for m in history:
            who = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            if m["role"] == "assistant" and len(content) > 500:
                content = content[:500] + " …"      # answers can be long → truncate to save tokens (enough to resolve references)
            lines.append(f"{who}: {content}")
        hist = "\n".join(lines)
    prompt = CONDENSE_PROMPT_TMPL.format(history=hist, question=question)
    rewritten = model_client.chat([{"role": "user", "content": prompt}], max_tokens=200).strip()
    return rewritten or question


def main():
    history = []
    while True:
        try:
            question = input("\nQuestion (or type v for a voice question): ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if question.lower() in ["exit", "quit"]:
            break
        if question.strip().lower() in ("v", "voice"):     # voice question: record → Whisper transcription
            try:
                import voice
                question = voice.listen()
                print(f"🗣  You said: {question}")
            except Exception as e:
                print(f"Voice unavailable (needs Apple Silicon + mlx-whisper): {e}")
                continue
        if not question.strip():
            continue

        # Multi-turn: first rewrite a follow-up (with references) into a standalone question,
        # then retrieve and answer (on the first question history is empty → returned as-is).
        standalone = condense_question(question, history)
        if standalone != question:
            print(f"[Condensed into English search query → {standalone}]")
        answer, sources = map_reduce(standalone)
        if not sources:
            print("\nNo relevant documents found.")
            continue

        print("\n" + "=" * 80)
        print(answer)

        print("\nSources:")
        for i, hit in enumerate(sources, 1):
            section = hit.get("section", "").strip()
            chunk_type = hit.get("type", "text")
            section_str = f"  [{section}]" if section else ""
            if chunk_type == "figure":
                section_str += "  [figure]"
            elif chunk_type == "table":
                section_str += "  [table]"
            snippet = hit["text"][:120].replace("\n", " ").strip()
            print(
                f"[{i}] {hit['paper']}  p.{hit['page']}{section_str}"
                f"  (rerank={hit.get('rerank_score', 0):.3f})"
            )
            print(f"     {snippet}...")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()
