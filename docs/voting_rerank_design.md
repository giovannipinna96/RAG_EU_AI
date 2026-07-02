# Multi-chunk voting rerank — design note

Status: **implemented and ON by default** (`ENABLE_CHUNK_VOTING`).

Motivated by job 54885: the adaptive-snippet fix still failed comp_2/comp_11 —
comp_11 showed the stock `5,4,3,2,1` rerank reversal, confirming the answering
chunk of Article 50 never reached the candidate pool because the pre-rerank
dedup kept the wrong Article-50 chunk. Code lives in
`TripleRetriever._final_rerank` / `_llm_rerank_vote` / `_aggregate_chunk_votes`;
tests in `tests/test_chunk_voting.py`.

**A/B result (30-question eval, jobs 56944 off vs 56945 on):**

| | voting OFF | voting ON |
|---|---|---|
| PASS | 26/30 | **28/30** |
| Ref recall | 85% | **92%** |
| Keyword recall | 78% | 76% |
| Avg latency | 1.91s | 2.42s |

Voting recovered comp_11 (deepfake → `Article 50.2`) and comp_1 (regression),
at +0.5s latency. The two remaining fails are **not** voting-addressable:
comp_2 is a genuine relevance-judgment ambiguity (Art 5 is in candidates but
the LLM prefers Annex III/Art 50), and comp_27 is a BGE cross-encoder demotion
(Art 62 is retrieved RRF #1 but BGE buries it to #8 — addressed by the
pool-blend follow-up below, not by voting). Re-run the A/B any time with:

```bash
sbatch --export=ALL,KEEP_API=0,ENABLE_CHUNK_VOTING=false slurm/sglang_eagle.slurm  # baseline
sbatch --export=ALL,KEEP_API=0 slurm/sglang_eagle.slurm                            # default (on)
```

## Follow-up: pool-blend for the comp_27 BGE-demotion (designed + implemented, A/B pending)

The voting A/B left comp_27 (gold Article 62) failing. The retrieval diagnostic
(job 57181) showed this is **not** a recall miss: Art 62 is retrieved well
(raw-dense #2, bm25 #1, **RRF-merged #1**) but the BGE cross-encoder demotes it
to **#8 (score 0.042)**. In voting mode the pool is selected by **pure BGE**
(`reranker.rerank(query, candidates, top_k=rerank_candidate_chunks)`), so once
the candidate set is un-deduped (many chunks per article) Art 62's best chunk
falls outside the top-`N` BGE pool and never reaches the LLM vote.

**Fix (`TripleRetriever._blend_pool`, flag `enable_pool_blend`, default off):**
reserve the top `pool_rrf_reserve` RRF candidates in the pool unconditionally,
then fill the remaining slots in BGE order. A doc every other retriever ranked
first is no longer silently dropped by one low cross-encoder score. Identity is
by `id()` (BGE returns the same dict objects). Tests in
`tests/test_chunk_voting.py` (`test_blend_pool_*` +
`test_final_rerank_blend_recovers_strong_rrf_article` and its pure-BGE
companion that shows the drop without the blend).

**Still to do:** A/B on the 30-question eval (flag off vs on) before defaulting
it on — same regression-risk gate as voting itself. Run with:

```bash
sbatch --export=ALL,KEEP_API=0 slurm/sglang_eagle.slurm                       # voting on, blend off (baseline)
sbatch --export=ALL,KEEP_API=0,ENABLE_POOL_BLEND=true slurm/sglang_eagle.slurm  # blend on
```

## Problem this solves

The current pipeline deduplicates candidates **by `article_id` before the
rerank** (`TripleRetriever.retrieve`):

```
sources → RRF merge → dedup-by-article_id → BGE rerank → LLM rerank → top-5
```

Dedup keeps, per article, only the single chunk with the highest fused RRF
score. That chunk is not necessarily the one stating the rule the question
asks about. Observed in the retrieval diagnostic for comp_11 (deepfake):

- BM25 returned 5 chunks of Article 50; dense returned more.
- The surviving (highest-RRF) chunk was **Art 50(1)** (chatbots), not
  **Art 50(4)** (deepfake disclosure).
- The answering paragraph was therefore never shown to the reranker — no
  reranker could recover it.

The adaptive-snippet fix (centering the LLM-rerank window on the query match)
mitigates this *when the answering clause is inside the surviving chunk*. It
does **not** help when the surviving chunk is the wrong paragraph of a
multi-chunk article.

## Proposed design

Move dedup to **after** the rerank and aggregate per article by voting:

```
sources → RRF merge → keep top-N chunks (NO article dedup)   [N≈12]
        → adaptive-snippet per chunk
        → LLM rerank all N chunks (single prompt)
        → aggregate by article_id:
              score(article) = Σ over its chunks  1 / (k + llm_rank_of_chunk)
        → sort articles by score → take top-5 articles
        → for each winning article, surface its best-ranked chunk as the citation
```

This is an RRF-style positional vote (`k` ≈ 10, reuse `settings.rrf_k`): an
article wins if *any* of its chunks ranks well, and several mediocre chunks can
still combine to beat a single strong-but-irrelevant one.

### Why voting over "just send more chunks"

Sending N un-deduped chunks to the LLM and taking the top-5 chunks directly
would let one article occupy multiple top-5 slots, starving other relevant
articles. Per-article aggregation guarantees 5 *distinct* articles in the
answer while still letting the best chunk per article carry the citation.

## Cost / latency

| | current | with voting |
|---|---|---|
| chunks in rerank prompt | ≤8 (deduped) | ≈12 |
| prompt size | ~24K chars (~6K tok) | ~32K chars (~8K tok) |
| LLM rerank calls/query | 1 | 1 (unchanged) |
| added latency | — | ~+200–400 ms (longer prompt) |
| extra code | — | ~+60 LOC + tests |

Still a single rerank call, so cost grows only with prompt length, not call
count. Fits a 32K-context utility model.

## Risks

- **Regression on the 16/18 (now ≥28/30) passing cases**: a longer, noisier
  prompt can perturb rankings that currently work. Gate behind a flag
  (`settings.enable_chunk_voting`, default off) and A/B on the full eval.
- **Index parse cost**: the rerank-response parser must map chunk indices back
  to articles; keep the chunk→article_id map alongside the prompt order.
- **Tie-breaking**: when two articles tie on vote score, fall back to the best
  single chunk rerank position.

## Implementation checklist (when greenlit)

1. `settings.enable_chunk_voting: bool = False`, `settings.rerank_candidate_chunks: int = 12`.
2. In `retrieve()`: when voting is on, skip the pre-rerank article dedup; pass
   top-N chunks to a new `_llm_rerank_vote(query, chunks, top_k)`.
3. `_llm_rerank_vote`: reuse `_build_rerank_prompt` + `_adaptive_snippet`;
   parse the ranking; aggregate `1/(k+rank)` per `article_id`; return one doc
   (best chunk) per winning article, sorted by aggregated score.
4. Tests: wrong-paragraph-of-right-article recovery; distinct-articles
   guarantee; tie-break; flag-off path identical to current behaviour.
5. A/B: run the 30-question eval with the flag off then on; compare ref recall
   and the per-question deltas before defaulting it on.
