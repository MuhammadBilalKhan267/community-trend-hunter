# Algorithms

[Architecture](ARCHITECTURE.md) · [Selection implementation](../trend_hunter/candidates.py) · [Temporal measurements](../trend_hunter/temporal.py) · [Evaluation](EVALUATION.md)

![Selection matrix: formulas, ranking order and review allocations](assets/selection-matrix.svg)

## 1. What counts as support?

For one candidate/topic, day `d` and platform `p`:

| Symbol | Definition | Unit |
| --- | --- | --- |
| $A_{dp}$ | Distinct platform/display-name author proxies | Author proxies |
| $A_{dpf}$ | Distinct author proxies posting normalized own-text family `f` | Author proxies per text family |
| $T_{dp}$ | Sum of each family's author count capped at three | Template-capped support |
| $q_{dp}$ | Smaller of author breadth and template-capped support | Aggregate capped support |
| $Q_R, Q_P$ | Sums of `q` in the recent and preceding seven-day windows | Support per window |
| $D_{14}, P_{14}$ | Active included days and represented platforms in those windows | Days; platforms |

$$
T_{dp}=\sum_f\min(3,A_{dpf}),\qquad
q_{dp}=\min(A_{dp},T_{dp})
$$

$$
Q_R=\sum_{d\in R,p}q_{dp},\qquad
Q_P=\sum_{d\in P,p}q_{dp},\qquad
Q_{14}=Q_R+Q_P
$$

Repeated posts by one author in the same family/platform/day count once. The author bound limits paraphrase-heavy posting; the family cap limits repeated templates across names. Adding observations cannot reduce `q`. These are aggregate bounds, not exact author-to-family allocations, per-author contribution guarantees or real-person identities. Raw activity remains available separately.

## 2. Activity ordering

The numeric `score` is **Q14**. Further features break ties rather than adding unlike quantities to the score.

$$
K(c)=(-Q_{14},-D_{14},-P_{14},-E,\operatorname{id})
$$

Sort ascending by this tuple: compare the first unequal value. `E` is engagement. Before semantic review it is nomination activity; for final topics, following-response counts require sampled attribution. Use `max(explicit replies, possible responses)` rather than adding overlapping streams.

| Order | Feature | Direction | Purpose |
| ---: | --- | --- | --- |
| 1 | Q14 | Larger first | Prefer substantive observed reach in the comparison horizon |
| 2 | Active window days | Larger first | Prefer recurrence when reach is equal |
| 3 | Window platform breadth | Larger first | Prefer broader observed distribution when reach and recurrence tie |
| 4 | Engagement | Larger first | Preserve discussion activity without inventing independent complaints |
| 5 | Stable ID | Ascending | Make equal cases reproducible |

**Candidate ordering:** prepend whether a supported temporal classification is absent. **Final ordering:** eligible normal-accepted topics precede all provisional fallback topics. The table applies within those tiers.

This is an explicit reach-priority policy: one additional support unit beats a later tie-break. It is not an estimated monetary impact or severity score.

The comparison horizon prevents old accumulation from dominating current discussion. Author breadth and template caps address repetition within the support quantity. Engagement remains separate because replies can overlap and do not establish independent experiences. Geometric cohesion guides grouping and sampling, not studio importance.

Raw volume, thread count, author count, all-history activity, cosine statistics, gaps, daily variation and image counts remain diagnostics.

## 3. Three review queues

| Queue | Ordering | First batch | Later batch |
| --- | --- | ---: | ---: |
| Overall | Supported temporal status, then `K` | 5 | 2 |
| New/growing | Normalized growth `G` descending, then recent capped support, then `K` | 3 | 1 |
| Unusual | Deterministic interleaving of fading, qualified bursts and remaining non-growing groups | 2 | 1 |

A missing comparable growth ratio sorts behind a measured ratio. The new/growing queue refers to activity direction, including complaints. Overall positions can contain these topics too; duplicate candidate IDs never consume two positions.

Within unusual review, fading sorts by strongest decline. Discussion bursts sort by their detector strength, template bursts by concentration/count, and other groups by activity. Discussion and template bursts alternate within their lane. Build the full interleaving before excluding reviewed IDs so later batches do not restart the same lane. Empty subqueues backfill.

The default 50% exploration share produces 5/3/2 for ten and 2/1/1 for four. Batch size and the exploration share are configurable. Policy reserves 60% of exploration for new/growing candidates before integer rounding.

## 4. Sampling and semantic admission

Let `n` be the number of examples in one packet, capped at ten. Each has a distinct original ID and focal text; all go into one logical Jev candidate request. Selection combines central, earliest, latest, boundary, platform-varied and reproducible additional examples.

$$
\text{normal admission: } m\ge\lceil0.8n\rceil
$$

The relevance and supported-scope decisions must also pass. With ten examples, eight must match. Smaller samples use their actual `n`. Matching dissent remains relevant membership; approval is not required.

| Sample result | Default route |
| --- | --- |
| At least 80%, relevant and scope-supported | Normal acceptance |
| At least 50% but below 80%, relevant and scope-supported | Provisional: five to seven matches out of ten; refine/watchlist |
| At most 20%, or irrelevant/background | Reject the interpretation |
| Intermediate/uncertain result | Mixed; bounded refinement if available |

One refinement inspects unsampled originals. Votes accumulate by focal text and reduce to one membership vote per original, so a topic's reviewed-source count can exceed ten. Admission uses the resulting matching-source ratio.

A provisional finding may fill a remaining vacancy only after candidate/review/preferred-token exhaustion, subject to available resource limits and successful writing checks. Normal accepted findings always come first. Provisional quality is explicit in the audit and readable explanation. Successful eight-result finalization does not rerun a provisional competition.

## 5. Temporal classification

`R` and `P` are consecutive seven-day calendar windows over eligible observed days. A possibly incomplete terminal day is excluded; missing observations are not silence. In this section, support means one platform/display-name/day, without the ranking template cap used in Q14.

For platform `p`, let `a` be topic author-day support and `u` the eligible universe's author-day support. Compare the same platforms with observations in both windows:

$$
S_W=\frac1{|\mathcal P|}\sum_{p\in\mathcal P}
\frac{a_{Wp}+0.5}{u_{Wp}+1},\qquad G=S_R/S_P
$$

Let $r_W=\text{author-day support}_W/\text{observed days}_W$. Share is platform-normalized; rate checks actual direction. Equal platform weighting limits collection-mix effects but does not estimate a population-weighted player share.

| Status | Main gates |
| --- | --- |
| **Growing** | `G >= 1.35`, `rR > rP`, at least five recent and three prior support units, two recent active days, and three observed days in each window |
| **Fading** | `G <= 0.65`, `rR < rP`, at least six prior support units, three active days and three observed days in each window |
| **New** | First appearance in the recent window, at least three recent support units over two days, no prior support, and three earlier observed days |
| **One-off** | Peak at least four support units and 60% of observed support; at most three active days; at least two follow-up observed days with mean tail at most 25% of peak |
| **Not real** | Apparent-amplification pattern below; this is not proof of fake accounts |

The bounded popular-discussion route also permits a recently observed substantive anchor, first matched within seven days, to qualify as **new** when sampled responses support attribution. This can qualify without the ordinary three-support/two-day gates. Responses establish engagement; they do not become independent substantive reports.

Precedence is apparent amplification, fading/growing/new, then a one-off shape. A relevant stable topic remains on the watchlist because the public schema has no stable status.

The normalized-share threshold supplies the effect-size requirement; the rate checks direction. A collapsing collection denominator alone must not create growing activity. The smoothed daily ratio remains a diagnostic.

### Repetition and collection boundaries

Apparent amplification requires at least twelve messages; at least 85% same-template concentration in a fifteen-minute interval and in one day; few distinct families (`<= max(2, floor(0.15 x count))`); and at most two distinct non-dominant platform/day/family combinations outside the interval. This nomination/classification cannot identify account age or establish bots.

Terminal truncation is suspected only with at least five preceding observed days, a preceding median of at least twenty eligible messages, terminal volume below 20% of that median, and the latest timestamp before 18:00. The fourteen-day lookback and uncertainty remain explicit. Observed source evidence is preserved even when that day is excluded from comparisons.

## 6. Conversation-burst nomination

An anchor must be nontrivial, not a short response, and have at least eight normalized text characters. Observe its next ten minutes in the same channel. Let `x` be short responses, `a` their distinct author proxies and `c` other non-short messages of at least eight characters. Let `b` be the expected ten-minute response count estimated from up to one preceding hour, using a time denominator of at least ten minutes.

$$
L=\frac{x+1}{b+1},\qquad
B=\frac{\ln(1+x)\ln(1+a)\min(4,L)}{1+c}
$$

Nominate only with at least four responses, two author proxies, `L >= 1.5` and at least 35% short responses in the interval. Pseudocounts prevent a zero background from producing an infinite lift. The lift cap limits a quiet-channel amplification effect. Competing anchors reduce attribution priority.

`B` orders discussion nominations against other discussion nominations. It is not added to Q14 or used as proof that every response refers to the anchor. If an anchor maps unambiguously to one cluster, enrich that cluster's review packet; otherwise keep a separate nomination. No model call occurs until the nominated packet is selected.

## 7. Reconciliation and public evidence

| Operation | Algorithm / limit | Why it remains |
| --- | --- | --- |
| Existing-topic comparison | Up to three current matching originals per topic inside the candidate request | Cheap incremental duplicate guidance |
| Neighbor nomination | Nearest eight centers; strong overlap can also qualify | Find separated branches outside the current shortlist |
| Strong overlap | At least three shared IDs and Jaccard `>=0.2` or half the smaller set | Avoid routing on one incidental shared source |
| Neighbor allowance | Sixteen review attempts per run; two per shortlisted topic | Bound review cost and prevent one topic consuming it |
| Final-set review | At most eight topics x ten reviewed originals; all pairs in one shared request | Reconsider identity collectively after incremental decisions |
| Merge groups | Every pair confirmed at the choice-confidence policy | Prevent unsupported transitive merges |
| Source recount | Set union, focal-ledger reduction and fresh history | Avoid duplicate support and stale status |
| Public evidence | Up to ten reviewed matching originals | Geometric membership alone never becomes an output `voice_id` |

Final source selection preserves anchor, early/latest and dissenting evidence where available, then adds date/platform/author variety. Context and aliases clarify those originals; they do not independently establish more reports.

## 8. Defaults and ownership

| Parameter / owner | Default | Purpose |
| --- | --- | --- |
| Cluster target / `discovery.py` | 80 distinct inputs | Computational granularity, not semantic topic size |
| Cluster bounds | 8-384; singleton handling below eight | Bound resource use; the ceiling produces larger groups on larger corpora |
| K-means computation | Batch 1,024; three starts; 100 iterations | Bound optimization work |
| Reproducibility / CPU | Seed 42; four clustering threads | Reproducible comparisons and bounded host contention |
| Template cap | Three per family/platform/day | Limit repeated-template support |
| Queue allocation | 50% exploration; 60% of that to new/growing | Review 5/3/2 initially and 2/1/1 in later batches |
| Context adjacency | Three preceding messages; 120 seconds | Different bounds on count and elapsed conversation time |
| Explicit context | Up to three ancestors and three replies per target | Bound graph expansion; preserve known relationships |
| Shared context | Up to twenty records; 18,000-byte JSON packet target | Remove context until the packet fits or context is empty; preserve intact target originals |
| Choice confidence | 0.55 | Retain uncertain outcomes rather than forcing membership/merge choices |
| Alias, release, attribution and wording support | 0.75 | Require stronger support for asserted relationships and labels; not calibrated truth probability |
| Candidate refinement | One additional sample | Add new evidence once, not repeat the same packet indefinitely |
| Semantic writing repair | One retry: two attempts | Bound repair cost; keep exact failures and allow backfill |
| Ollama schema repair | Two local attempts | Recover malformed output separately from semantic repair; invalid hosted output is recorded without automatic retry |
| Final generated fields | Chinese title, English title, proposed action | Code owns category, sources, dates, activity and `why` composition |
| Generation lengths | Titles 140 characters; action 300; final max 1,800 tokens | Schema bounds prose; the selected provider receives the output-token ceiling |
| Ollama generation context / batching | 16,384 / 64 | Bound host-memory use |
| Embedding batching / residency | 64 / 2,048 inputs; zero residency disables periodic release | Bound request memory and continuous model loading |
| Timeouts | Local detector 600s; ingestion 300s; Jev 60s; OpenAI writer 180s | Operational failure bounds, not quality weights |
| Provider context recovery | One default-to-long transition after an exact `max_tokens_exceeded` error | Preserve the full payload; separate cache mode and per-POST accounting. Shared transient retry allowance remains bounded. |
| Review count | Unset, or configured e.g. 100 | Stop at eight checked outputs, exhaustion or resource limits; no hard-coded review-count ceiling |
| Jev allowance | Preferred 20M; maximum 50M accounted tokens | Persistent reported usage plus unknown-cost reservations |

Operational overrides are documented by `python run.py --help`. Numeric bounds are explicit defaults, not empirically optimal settings or calibrated probabilities. Topic names and required example IDs do not determine these values.

## 9. Verification scope

Property tests cover format-invariant ordering, bounded and monotone support, capped repetition, queue routes, accepted-first fallback, evidence membership, cutoff isolation and context preservation.

Sample membership is not a verified census, and software tests do not prove Chinese understanding. The [evaluation](EVALUATION.md) separates measured software behavior, bounded source interpretations and remaining failures.
