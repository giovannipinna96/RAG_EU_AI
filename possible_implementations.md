# Possible Implementations — Linguistic Heuristics for Query-Complexity Detection

> Survey of query-complexity / query-difficulty heuristics for the EU AI Act RAG
> system, with a ranked recommendation for **this** project. Written 2026-05-27.
>
> Context: `src/retrieval/query_engine.py::_detect_complexity` decides whether a
> query is *complex* (→ decompose into sub-queries, expand retrieval) or *simple*
> (→ direct answer). Today it uses a **two-level cascade**: Level 1 = a 0 ms
> lexical signal count (`COMPLEXITY_SIGNALS`), Level 2 = an LLM SIMPLE/COMPLEX
> classification only for the uncertain middle band.

---

## TL;DR

- **Complexity detection is NOT the current quality bottleneck.** Eval (job 54744)
  routes all 18 questions correctly; the two failures (comp_2, comp_11) are
  retrieval/grounding misses, not misrouting. So the goal is a **zero-latency
  robustness upgrade**, not a research project.
- **Do:** Tier 1 — weighted scoring + AI-Act-specific structural signals
  (especially conditional/exception framing). Pure Python, 0 ms, no new deps.
- **Maybe later:** Tier 2 — one spaCy syntactic signal (subordinate-clause /
  dependency-distance), only if the LLM fallback still fires too often.
- **Skip:** Tier 3 — full QPP predictors, parse-tree batteries, learned
  classifier. Negligible RAG correlation and/or training data we don't have.

---

## The four families of linguistic heuristics

### 1. Lexical / structural surface heuristics  *(what we already have)*
Count cue words and surface markers:
- coordinating conjunctions (`or`, `and`)
- comparatives (`versus`, `compared to`, `differ`)
- causal connectives (`because`, `caused by`, `resulted in`)
- multiple `?`
- multiple named entities / references (≥2 Articles/Annexes)

Zero-cost, no dependencies. This is the `COMPLEXITY_SIGNALS` list in
`query_engine.py:33`. Weakness: every signal is equal-weighted and AI-Act legal
structure (exceptions, derogations) is invisible to a generic word list.

### 2. Pre-retrieval QPP predictors (IDF / SCQ / clarity / VAR)
Classic Query Performance Prediction:
- `avgIDF` / `maxIDF` — rare terms ⇒ more specific query
- `SCQ` — sum of tf-idf of query terms vs the collection
- `SCS` / clarity — KL divergence of query LM vs collection LM
- `VAR` — variance of term weights

**Verdict for RAG: not worth it.** Recent RAG-focused work reports pre-retrieval
QPP has *"low to negligible correlation"* with end-to-end answer quality on
neural retrieval; post-retrieval predictors do better but require a retrieval
pass first (latency). Also needs full collection-LM statistics.

### 3. Syntactic / parse-based complexity (dependency features)
- parse-tree depth
- dependency distance (avg distance between syntactically linked words)
- embedded-clause / subordinate-clause count
- Yngve scores (working-memory load)
- long-distance dependency ratio

Strong correlation with human-judged comprehension difficulty, but: (a) needs a
parser (spaCy), and (b) measures *reading* difficulty, not *retrieval-hop*
difficulty — correlated, not identical. The single most predictive feature is
subordinate-clause / dependency distance.

### 4. Learned complexity classifiers  *(the SOTA route)*
- **Adaptive-RAG** (Jeong et al., NAACL 2024): a T5-Large classifier routes each
  query into **A = no-retrieval / B = single-hop / C = multi-hop**, trained on
  auto-labeled data (single-hop: SQuAD / NQ / TriviaQA; multi-hop: MuSiQue /
  HotpotQA / 2WikiMultiHop). Reports up to **+39% answer correctness**; newer
  routing variants cut **latency ~35%**.
- **HANRAG / PRISM (2025)**: concatenate *semantic embeddings + explicit
  linguistic features* and predict a **hop count** used as the retrieval
  condition.

**Verdict for this project: overkill.** Needs labeled training data we don't
have and an extra model to serve, for an 18-question competition where an LLM
fallback already exists.

---

## Ranked recommendation for THIS project

### ⭐ Tier 1 — DO THIS: weighted scoring + domain-aware signals (0 ms, no deps)
Upgrade `_detect_complexity` from equal-weight `score>=2` to weighted scoring,
and add AI-Act-specific structural signals:

| Signal | Examples | Weight |
|---|---|---|
| Comparative / contrastive | `differ`, `versus`, `compared to`, `whereas`, `unlike` | **2** |
| Conditional / exception (legal!) | `unless`, `except`, `provided that`, `derogation`, `exemption`, `does … also apply` | **1.5** |
| Multiple distinct refs (≥2 Articles/Annexes) | "Article 6 and Annex III" | **2** |
| Coordination of obligations | `and` / `or` joining noun phrases | **1** |
| Conjoined questions (≥2 `?`) | "Is it prohibited? Or high-risk?" | **2** |
| Length gate (>30 tokens in resolved query) | long multi-clause query | **1** |

Routing: `score==0 → simple`; `0 < score < 2.5 → LLM fallback`; `score≥2.5 → complex`.

Rationale: the conditional/exception family is the genuinely novel, domain-fit
part — it captures the comp_2/comp_3 "always prohibited? / exception" shape that
a generic word list misses. Defensible in the write-up as "domain-aware query
routing grounded in the QPP / Adaptive-RAG literature."

### Tier 2 — OPTIONAL: one spaCy syntactic signal
If Tier 1 still leaves the LLM fallback firing too often, add **just**
subordinate-clause / dependency-distance count from a spaCy `en_core_web_sm`
parse, folded in as one more weighted signal (deep subordination → multi-hop).
~5 ms/query. Cost: pulls in spaCy + a model download (SLURM/Apptainer step) on
top of the already-heavy `ml` extra. Only if Tier 1 is empirically insufficient.

### Tier 3 — SKIP (document as "considered, rejected")
Full QPP, parse-tree-depth batteries, learned classifier. Keep them in the
write-up to show the field was surveyed.

---

## References

- Adaptive-RAG: Learning to Adapt RAG through Question Complexity (Jeong et al., NAACL 2024) —
  https://www.researchgate.net/publication/382632436_Adaptive-RAG_Learning_to_Adapt_Retrieval-Augmented_Large_Language_Models_through_Question_Complexity
- RAG Performance Prediction for QA (pre- vs post-retrieval QPP) —
  https://arxiv.org/html/2604.07985v1
- On the Limitations of Query Performance Prediction for Neural IR —
  https://ceur-ws.org/Vol-3478/paper04.pdf
- Can QPP Choose the Right Query Variant? (QPP for RAG pipelines) —
  https://arxiv.org/html/2604.22661v1
- HANRAG: Heuristic Noise-resistant RAG for Multi-hop QA —
  https://arxiv.org/pdf/2509.09713
- PRISM: Agentic Retrieval with LLMs for Multi-hop QA —
  https://arxiv.org/pdf/2510.14278
- Question Decomposition for RAG (ACL 2025) —
  https://aclanthology.org/2025.acl-srw.32.pdf
- Syntactic Dependency Distance as Sentence Complexity Measure —
  https://www.researchgate.net/publication/266584664_Syntactic_Dependency_Distance_as_Sentence_Complexity_Measure
- Textual form features for text readability assessment (Cambridge NLP) —
  https://www.cambridge.org/core/journals/natural-language-processing/article/textual-form-features-for-text-readability-assessment/08B54744EFD8327FC835DA730F8AC9BB
