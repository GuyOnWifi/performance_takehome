# Task 1 — Benchmark design report

**Status: draft, without MI355X data.** The suite is built and has run on real LLMD, but only with
Qwen3-0.6B on an RTX 3060 using the CUDA image. No result here is a performance claim about
Qwen3.6-27B on the MI355X. `workload.json` is not frozen: arrival rates, the reference and
`max_anchor_divergence` are still null, so a run of the real plan can't yet be `valid`. Sections
marked **Pending** list what the pilot must supply.

## Evaluation goals and workloads

The benchmark answers three separate questions: how long until the first token, how smoothly the
rest arrives, and how long a fresh server takes to produce a valid generation. There is no blended
score, because a fast server with stalls and a slow server with smooth output are different problems.

| Cell | Prompts | Output | Load | What it exposes |
| --- | --- | --- | --- | --- |
| `chat_light` | 256 tokens | Exactly 256 | 25% of capacity | Baseline TTFT and smoothness with no queueing |
| `bursty` | 256 tokens | Exactly 256 | 25% of capacity, clumped | Short peaks the average rate hides: admission, batching, prefill contention |
| `long_prompt` | 4096 tokens | Exactly 256 | 25% of capacity | Delayed prefill, which fast decode averages would hide |
| `long_output` | 128 tokens | Exactly 1024, `ignore_eos` | 25% of capacity | Decode stalls and use of slack |
| `near_capacity` | 80% short, 20% long | Exactly 256 | 75% of capacity | Queueing, drops and unfair scheduling |
| `prefix_hit` | 4096 tokens, shared prefix | Exactly 256 | 25% of capacity | Prefix-cache behavior |

Arrivals are seeded Poisson, because bursts are what real clients send, and a closed loop would hide
queueing by slowing down with the server. `bursty` is `chat_light` with gamma gaps of shape 0.25
(vLLM's `--burstiness`): the same average rate, but the gaps' coefficient of variation is 2 instead of
1, so requests arrive in clumps with long quiet spells between. Only the arrival shape differs from
`chat_light`, so a gap between the two cells is the cost of bursts alone. A Gaussian would be the
wrong shape here: gaps can't be negative, and real traffic has a long tail of quiet spells. It is in
full mode only, to keep quick mode's cost unchanged. Rates are set as a share of measured capacity, so they
remain meaningful on any hardware. Every cell uses `ignore_eos`, so output length is set by the plan,
not by when the model chooses to stop. Without it, a model or engine change that writes shorter
answers would look like a speedup, and two runs wouldn't do the same work. A request that stops before
`max_tokens` is `truncated_output`, which also catches a server that ignores the flag. Canaries,
anchors and startup probes keep natural stopping, because their answers are read from the text.

Prompts are real text, not random token IDs, because speculative decoding's acceptance rate depends
on content, and random input would understate a server that drafts. ShareGPT was considered and
rejected: it can't fill 4096-token prompts, its licence is unclear, and it would add a large download
to reproduction. Instead, `collect/corpus.md` holds 36 original passages written for this suite, of
eleven kinds: emails, stories, explainers, code, meeting notes, policies, incident reports, product
docs, recipes, essays and reviews. A short prompt is one passage followed by a question that fits its
kind ("Draft a reply to the email above."). A long prompt is a set of passages pasted in as documents,
the way a retrieval system builds one, followed by a task over all of them. Passages are cut at the
last sentence or line break inside the token budget, so prompts read naturally and are slightly
shorter than nominal: 200–254 tokens for short, 4027–4094 for long. With only 36 passages, short
prompts repeat content under different questions; the unique session line still makes every request
a cache miss. The corpus is small and written in one voice, so drafts may predict it better than
real traffic, and a speculating engine may look somewhat better than it would in production.

Omitted: multi-turn conversations, tool calls, multi-GPU serving, and prompts over 4096 tokens. `cold_fs` needs host rights to drop the page cache, so results report `warm_fs` only.

**Pending:** capacity on the MI355X, which sets every `rate_rps`.

## Metrics and validity

**TTFT** runs from the send time to the first event carrying generated text, on either the reasoning
or content channel. First-content TTFT is reported as well. Role-only and empty events never count.
A failed request stays in the population, censored at `request_timeout_s`, so failing slow requests
can only make the result worse.

**Fluidity and FTGR** use the shared Etalon code on token times, not chunk times. Streamed text is
retokenized with the pinned tokenizer, and each token gets the receive time of the event that
completed it. Tokens sharing an event share a time, so bursts stay visible and slack decides whether
they hurt. A failed request scores fluidity 0.

**TPOT and TPS** are diagnostics only. Fluidity already judges streaming, so they're never gated or
compared, and they can't change a result. They're reported per cell to help explain one: TPOT and
request TPS at each reported quantile, plus system TPS.

**Startup** runs from launch start to the first probe that produces a valid completion. A readiness
response never stops the clock.

`suite/STATES.md` is the complete mapping. In summary:

| State | Meaning | Examples |
| --- | --- | --- |
| `valid` | The evidence is complete and can be believed | All gates pass |
| `invalid` | The server broke its contract or the evidence is corrupt | Wrong canary, false usage, failures over 1%, crash, no valid startup, corrupt timestamps, vanished requests |
| `inconclusive` | Nothing shows a violation, but the evidence is too weak | Too few requests for a quantile, client lag, too few anchors, no reference, unfinished warmup |

`invalid` beats `inconclusive`, which beats `valid`, at every level. Client-observed times and bytes
are trusted as the measurement. Everything the server claims is checked independently.

**Correctness uses two checks.** Canaries are seeded arithmetic prompts, passing when the last number
is the answer. Anchors are fixed prompts compared with a reference generated per model revision.

Real LLMD changed five rules that the fake server had let pass:

| Original rule | What real LLMD did | Rule now |
| --- | --- | --- |
| Canary text equals the answer | 0 of 120 passed, because the model shows its working | The last number must match; canaries the model gets wrong in the reference run are never sent |
| Usage equals the streamed token count | `usage` includes `<think>` tags that are never streamed | Hidden tags are added to the count |
| Usage within 1 token | Retokenizing fell short by up to 3 of 256 tokens | Tolerance is the larger of 1 token or 2%, applied to usage and to every length floor |
| Anchor text equals the reference | Fresh sequential: 24 of 24 matched. With or after concurrent traffic: 7 to 19 of 24 | First 8 characters, finish reason and at least half the reference length |
| Canaries get 64 tokens | On the MI355X with Qwen3.6-27B, LLMD ignored both `/no_think` and `enable_thinking: false`; all 120 canaries were still reasoning at token 64, so the reference kept none | 1024 tokens (the longest correct canary used 577); reference canaries run 16 at a time |

Because the anchor rule is weaker, a server returning plausible wrong text with the right opening and
length passes anchors. Canaries are the remaining defense, and they only show that the server
answers arithmetic correctly under load. Reasoning output often opens the same way, so the
8-character check mostly catches placeholders. A server that generates everything, then releases
tokens at a steady pace, can't be told apart from true streaming, though its cost appears in TTFT.

Fail-fast stops capture only when a run is already certain to be `invalid`: a crash, a failed
startup, three run-failing requests, or failures in one cell above 1% of that cell's *planned*
requests. An earlier rule stopped at 5% of the first 50, which a large cell could still recover
from, so it was replaced. Healthy runs always finish,
so there is no bias from stopping on good results. `false_usage` can't trigger it, because capture
doesn't tokenize during a run. Such a run is still `invalid`, but costs its full duration.

**Pending:** re-measure all four rules on Qwen3.6-27B.

## Statistical policy

**Warmup and cache:** a launch warms up with 20 requests that are excluded from metrics. Cells report
their prefix-cache state and states are never pooled. Unique prefixes force a miss, and a shared prefix
tests a hit. Canaries and anchors repeat exact bytes, so they're excluded from TTFT and fluidity, but
they count towards failure rates.

**Repetitions** are the independent unit, because requests in one repetition share queue and server
state. Repetitions are spread across launches in a seeded order, so that launch effects don't stack up
on one cell.

| | Quick | Full |
| --- | --- | --- |
| Cells | 2 | 6 |
| Repetitions per cell | 5 | 8 |
| Window | 120 s, or until 53 counted requests | 120 s, or until 53 counted requests |
| Launches | 3 | 4 |
| Quantiles | p50, p90 | p50, p90, p99 |

**Quantiles** are reported only when `n(1-q) >= 5`; below that a quantile is close to the maximum.
Only the per-repetition p90 gates a cell, because it's what comparisons use. Each repetition needs 50
counted requests, and a cell with fewer is `quantile_underpowered`. p99 needs 500 requests per cell. It
is reported when it has them and left empty otherwise, without changing the cell's state.

**Windows** run for 120 s, and keep going past that until they hold 53 counted requests: the 50 that
p90 needs, plus 5% spare. Canaries and anchors don't count. The schedule is the same seeded sequence,
so a longer window only extends it. This rule came from the MI355X full run. Rates follow measured
capacity, so a slow cell on slow hardware gets few requests in a fixed window, and three of the six
cells there had 22 to 51 per repetition.

A plan can also name extra modes with the same fields. The targeted MI355X re-run used one, `fill`,
covering only the cells that needed it.

**Comparison** uses log p90 per repetition, mean fluidity per repetition and log startup per launch.
Each gets a Welch t-interval with a Bonferroni adjustment across the mode's primary comparisons. A
bootstrap was rejected because 5 to 8 values give it too few distinct resamples. Each side needs at
least 3 repetitions.

| Decision | Rule |
| --- | --- |
| `regression` / `improvement` | The whole interval is past the margin |
| `equivalent` | The whole interval is inside the margin |
| `inconclusive` | The interval crosses a margin, or evidence is too weak on either side |
| `invalid` | Either run is `invalid` |

Margins are provisional: 0.1 log for TTFT and startup, and 0.02 for fluidity.

An `inconclusive` run makes every decision against it `inconclusive` (`run_inconclusive`), so one
underpowered cell blocks the whole comparison. That's why the window rule above matters.

**What the margins can resolve on the MI355X.** The full run's spread between repetitions sets how wide
an interval two runs of that spread would get, using the suite's own interval code. The rows below give
the smallest TTFT or startup slowdown that could be called a `regression`, which is the margin plus
that half-width. They also say whether `equivalent` is reachable at all: it needs a half-width under the
margin. Quick mode uses 5 repetitions per side and full mode 8; startup uses 3 and 4 launches.

| Cell / axis | SD per repetition (log) | Smallest regression, quick / full | `equivalent` reachable |
| --- | --- | --- | --- |
| `chat_light` TTFT | 0.039 | 20% / 18% | Yes |
| `long_output` TTFT | 0.045 | 22% / 19% | Yes |
| `prefix_hit` TTFT | 0.043 | 21% / 19% | Yes |
| `bursty` TTFT | 0.095 | 35% / 30% | No |
| `long_prompt` TTFT | 0.114 | 41% / 35% | No |
| `near_capacity` TTFT | 0.422 | 171% / 129% | No |
| `near_capacity` fluidity | 0.019 | +0.060 / +0.053 | No |
| Startup | 0.040 | 28% / 26% | No |

Fluidity on the other cells is close to 1.000 in every repetition, so its 0.02 margin is what decides.
Log TTFT residuals by launch averaged between −0.04 and +0.03, within their standard errors, so launches
add no visible effect. `near_capacity` is the outlier: at 75% of capacity, queueing makes its p90 vary
by about 50% between repetitions, so it can only flag large changes. Startup is dominated by a ~28 s
compile, so its spread is small, but 3 or 4 launches give a wide interval. The spreads for
`long_prompt`, `long_output` and `prefix_hit` come from underpowered repetitions, so they are rough.

Identical runs show that the unpaired interval is too wide. Two A/A runs differed by only 0.002 and
0.003 log on TTFT, yet both came back `inconclusive`, because the interval reflects different schedules
across repetitions, not disagreement between runs. Comparing matching repetitions as pairs should
tighten this, but it needs real A/A data before being adopted.

**Pending:** margins, deadlines and `max_anchor_divergence` from MI355X A/A runs, and the paired test.

## Validation matrix

Everything below is synthetic or debug evidence, except the two MI355X runs at the end.

**Synthetic faults:** 47 families land in their expected state with 0 mismatches
(`suite/tests/matrix.py`). They cover bursts, stalls, late first tokens, reasoning channels, Unicode
boundaries, missing completion, errors and drops, invalid output, false usage, misleading readiness,
and clean and noisy controls. The families read behavior only and never see a label.

**False positives:** 150 A/A decisions in each of the quiet and noisy controls over 30 synthetic pairs
in quick mode: 0 confident regressions.

**Detection:** quick mode, 30 synthetic pairs.

| Fault | Fault size → share detected |
| --- | --- |
| Slower TTFT | 5%: 0, 10%: 0, 20%: 97%, 50%: 100% |
| Stalled requests (fluidity) | 2%: 0, 5%: 0, 10%: 3%, 25%: 97% |
| Slower startup | 5%: 0, 10%: 0, 25%: 77%, 50%: 100% |
| Failed requests (run `invalid`) | 0.5%: 0, 1%: 77%, 2%: 100%, 5%: 100% |

Faults smaller than the margins are expected to go undetected, so a miss at 10% is not a false
negative. Those cases mostly return `inconclusive` or `equivalent`. Full-mode sensitivity hasn't been
simulated.

**Real LLMD:** Qwen3-0.6B on the RTX 3060, 3 launches, 2 cells, 3 repetitions of 60 s, 676 requests per
run. These runs used `artifacts/debug-cuda/workload.debug.json`, written by `tests/e2e/e2e_cuda.py`,
which sets values the committed plan leaves null or pins higher: `rate_rps` 2.0 and 1.5,
`max_anchor_divergence` 0.1 and `min_stable_anchors` 10. The committed `workload.json` would score
the same evidence `inconclusive` (`divergence_unmeasured`). `near_capacity` ran close to overload here, so its wide intervals say little about the design.

| Run | State | Failed | Anchors diverged | TTFT p90 `chat_light` | Against `clean_a` |
| --- | --- | --- | --- | --- | --- |
| `clean_a` | `valid` | 0 | 0 of 41 | 42 ms | Baseline |
| `clean_b` | `valid` | 0 | 0 of 41 | 42 ms | Fluidity `equivalent`; TTFT and startup `inconclusive`; no false regression |
| `--batch-size=1` | `valid` | 0 | 0 of 41 | 59.7 s | TTFT `regression` on both cells; fluidity `regression` on `chat_light`, `inconclusive` on `near_capacity`; startup `inconclusive` |

**After `ignore_eos` on every cell:** the same debug plan was re-run as an A/A pair, with evidence in
`tasks/benchmark-design/artifacts/debug-cuda-eos/`. LLMD honours the flag: a 200-token request
stopped at 119 tokens without it, and returned 200 with `finish_reason=length` with it. Both runs
were `valid`, with 676 of 676 requests completed, 0 failed and 0 truncated. Against `clean_a`, fluidity
on `chat_light` was `equivalent`, and every other decision was `inconclusive`, with no false
regression. TTFT p90 on `chat_light` was 41 and 43 ms. Each run took 411 s.

The degraded run is a real engine setting, not an injected fault. Startup took about 6 s and wasn't
expected to change.

**Fake server over real HTTP:** a clean run, wrong answers and a crash were re-run on the final code.
Wrong answers stopped at 48 s and a crash at 13 s, both `invalid`. False usage and slow TTFT last ran
before the out-of-scope checks were removed, when they were caught as `invalid` and `regression`.

**MI355X, one full-mode run:** `zmlai/llmd@sha256:0a2cd01a…` started with only `--io-impl=zio
--model=…`, Qwen3.6-27B, warm filesystem cache, in a Daytona on-demand sandbox. `collect/sweep.py` set
the rates: closed-loop capacity with 32 requests in flight, times each cell's load share. All 4062
planned requests completed and none failed, 1 of 207 anchor checks diverged, and startup was `valid`.
The run is `inconclusive` because three cells are `quantile_underpowered`.

| Cell | Capacity | Rate | Counted | TTFT p90 | Fluidity | FTGR | State |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `chat_light` | 3.33 rps | 0.831 rps | 690 | 116 ms (109–123) | 1.000 | 55.3 tok/s | `valid` |
| `bursty` | 3.33 rps | 0.831 rps | 805 | 181 ms (157–209) | 1.000 | 53.6 tok/s | `valid` |
| `long_prompt` | 1.38 rps | 0.346 rps | 270 | 1548 ms (1306–1836) | 0.963 | 19.7 tok/s | `inconclusive` |
| `long_output` | 0.90 rps | 0.225 rps | 213 | 62 ms (57–66) | 1.000 | 56.2 tok/s | `inconclusive` |
| `near_capacity` | 2.15 rps | 1.616 rps | 1407 | 2223 ms (1180–4186) | 0.968 | 14.9 tok/s | `valid` |
| `prefix_hit` | 1.33 rps | 0.333 rps | 300 | 1537 ms (1440–1640) | 0.976 | 21.6 tok/s | `inconclusive` |

Startup was 37.2 s (31.6–43.7) over 4 launches, ranging from 35.3 to 38.6 s.

What the cells showed:

- At the same mean rate as `chat_light`, bursts raised TTFT p90 by 56% and left fluidity unchanged.
- At 75% of capacity, queueing pushed TTFT p90 past 2 s, with a wide interval.
- `prefix_hit` and `long_prompt` send prompts of the same length. Their TTFT and capacity matched, so
  the shared prefix gave no measurable gain with LLMD's defaults. The image's `llmd --help` has no
  prefix-cache option, so the default is the only behaviour to measure.

**The underpowered cells were a design gap, now closed.** At 25% of capacity in 120 s windows,
`long_output` got 22–33 counted requests per repetition, `long_prompt` 28–43 and `prefix_hit` 30–51,
short of the 50 that p90 needs. Setting rates from capacity keeps load comparable across hardware, but
the sample count then falls with capacity, and a fixed window doesn't make up for it. The window rule
in the statistical policy fixes this. Quick mode's two cells had at least 75 and 159 per repetition
here, so they were powered either way.

**MI355X, targeted re-run:** the same three cells under the window rule, later the same day in a new
sandbox. It used the full run's plan, rates and reference, plus a `fill` mode: 2 repetitions per cell,
3 launches, p50 and p90. The offline schedule predicted 53 counted requests in every window and 1340 s
of requests, and the run matched both. All 354 requests completed and none failed, none of 20 anchor
checks diverged, and the run is `valid`.

| Cell | Window | TTFT p90 per repetition | Full run, 8 repetitions | Fluidity per repetition | State |
| --- | --- | --- | --- | --- | --- |
| `long_prompt` | 202–221 s | 1399, 1356 ms | 1360–1833 ms | 0.970, 0.976 | `valid` |
| `long_output` | 257–262 s | 62, 63 ms | 57–65 ms | 1.000, 1.000 | `valid` |
| `prefix_hit` | 181–217 s | 1468, 1582 ms | 1448–1673 ms | 0.978, 0.975 | `valid` |

Startup was 36.8, 36.5 and 36.5 s, inside the full run's 35.3–38.6 s.

What it does and doesn't show:

- Every repetition lands inside the full run's range, fluidity included, on a different sandbox. That
  is a consistency check, not an A/A. The plans differ, so the suite rejects the pair as
  `identity_mismatch`, and 2 repetitions are below the 3 a comparison needs.
- With 2 repetitions the t-interval has one degree of freedom, so the report's intervals are too wide
  to use, for example 334–5676 ms on `long_prompt`. The run was sized to show the cells can be
  powered within the remaining budget, not to narrow their intervals.
- `prefix_hit` was no faster than `long_prompt` here either, with pooled p50s of 1161 and 1173 ms.

There's no MI355X A/A yet, so `max_anchor_divergence` stays at the provisional 0.1 from the 3060. The
observed divergence was 0.5% in the full run and 0 of 20 in the re-run.

**Limitations:** only one real degradation has been tested. Real faults on the MI355X will be limited
to what can be produced without changing the engine, so most fault coverage stays synthetic. The real
A/A pair is a single sample, not a false-positive rate. Fake-server results are never validation
evidence.

**Pending:** A/A controls and real slowdowns on the MI355X.

## Measured compute cost

The debug plan on the 3060, and one full run on the MI355X:

| Run | Wall time | Launch | Warmup | Requests | Drain | Cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| `clean_a` | 403 s | 17 s | 18 s | 357 s | 8 s | 2 s |
| `clean_b` | 403 s | 18 s | 18 s | 357 s | 8 s | 2 s |
| `--batch-size=1` | 759 s | 17 s | 17 s | 357 s | 366 s | 2 s |
| MI355X full | 6265 s | 149 s | 98 s | 5675 s | 338 s | 3 s |
| MI355X targeted re-run | 1585 s | 110 s | 73 s | 1340 s | 60 s | 3 s |

The degraded run shows why drain is measured: overload doubled the run's cost, all of it after the
request windows ended.

The MI355X full run took 1.74 GPU-hours, inside the plan's 2.5 h limit, and the re-run 0.44. Each launch took about 37 s,
with the model in a warm page cache. New hardware also needs a one-off setup before its first run:
19 s to download the model, 844 s for the reference and 476 s for the capacity sweep. The reference
ran its canaries one at a time; they now run 16 at once. Using the same launch, warmup and drain
times, quick mode would take about 25 minutes, or 0.4 GPU-hours.

Planned request time, excluding every other phase:

| Mode | At 120 s windows | MI355X rates, with the window rule | Wall limit in plan |
| --- | --- | --- | --- |
| Quick | 2 × 5 × 120 s = 20 min | 20 min | 40 min |
| Full | 6 × 8 × 120 s = 96 min | 130 min | 2.5 h, inside the 3 h ceiling |

The MI355X column comes from the seeded schedule with the rates `collect/sweep.py` measured there. The
fast cells still fill 120 s windows. `long_prompt`, `long_output` and `prefix_hit` stretch to 143–287 s.
On the MI355X, launches, warmup and drain came to 10 minutes, so a full run there would take about 140
minutes, 10 minutes inside the plan's limit. Gating on p99 as well would have needed 150 minutes of
requests alone. Slower hardware stretches windows further, so the limit is what bounds a run;
`evaluate.py` flags a run that overruns it as `over_budget`. Repetitions were kept at 8 because they
set the interval width.

**Pending:** a measured quick run on the MI355X. One was started and stopped after 10 minutes, to put
the remaining budget into the targeted re-run. The quick figure above is a projection.

## Reproducibility

Everything uses the suite under `tasks/benchmark-design/suite/`. The debug runs predate the suite's
first commit, so their manifests record the starter, `81aead87d78ed468c4c374fb42921471b44c3f80`, and
don't identify the suite code that produced them.

| Path | Job |
| --- | --- |
| `evaluate.py` | Runs the analysis, writes `report.json`, prints the summary table |
| `workload.json` | The benchmark plan |
| `STATES.md` | Rules for `valid`, `invalid` and `inconclusive` |
| `requirements.txt` | Dependencies for capture only; `evaluate.py` is stdlib-only |
| `analysis/events.py` | Loads evidence and the plan; the only module that knows the event format |
| `analysis/gates.py` | Decides validity for requests and runs |
| `analysis/axes.py` | Computes TTFT, fluidity, FTGR and startup per cell |
| `analysis/compare.py` | Intervals and comparison decisions |
| `collect/capture.py` | Stand-in for the organizer's runner; also generates the reference |
| `collect/sweep.py` | Measures each cell's capacity on a running server and sets `rate_rps` |
| `collect/tokenize_events.py` | Independent token counts and times |
| `collect/corpus.md` | Original passages the prompts are built from |
| `collect/make_prompts.py` | Generates the seeded prompt sets |
| `prompts/` | Generated prompts |
| `schemas/` | Plan and event formats |
| `tests/synth.py` | Synthetic evidence with injected faults |
| `tests/matrix.py`, `tests/sensitivity.py` | Expected states, detection and false-alarm rates |
| `tests/test_*.py` | Unit tests |
| `tests/e2e/` | Full pipeline against a fake server and against real LLMD |

```bash
python tasks/benchmark-design/suite/tests/matrix.py
python -m unittest discover -s tasks/benchmark-design/suite/tests
python tasks/benchmark-design/suite/tests/sensitivity.py
python tasks/benchmark-design/suite/evaluate.py --events <dir> \
  --workload tasks/benchmark-design/suite/workload.json --out report.json
# Rates for new hardware, from suite/, against a running server
python collect/sweep.py --workload workload.json --base-url http://127.0.0.1:8000 \
  --out <dir> --write-plan workload.<gpu>.json
# Real LLMD on a CUDA GPU, from suite/, with requirements.txt installed
python tests/e2e/e2e_cuda.py --out <dir> --model-dir <Qwen3-0.6B directory>
```

| Identity | Value |
| --- | --- |
| `workload.json` SHA-256 | `c9470814d97e2cddf10b81e14a6b0ae13ecd67ff997f435804772e42754e0058` |
| Seed | 20261030 |
| Debug image | `zmlai/llmd@sha256:752fc2a8a50971a14d2035e653c6c8ddf672d257b3cfed7eb43f216a123bda7c` |
| Debug model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| MI355X image | `zmlai/llmd@sha256:0a2cd01a2e53c78934a8cfef66f8668da7666be4954a5d18b72df089c54ea34f` |
| MI355X model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Prompt tokenizer | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |

Raw debug evidence is in `tasks/benchmark-design/artifacts/debug-cuda/` (247 MB). The path is ignored
by git, so it needs separate delivery. Its reports were re-scored after the last code cut.

MI355X evidence is in `tasks/benchmark-design/artifacts/mi355x-full/` (720 MB, also ignored), with its
plan, reference, report and the scripts that ran it. The run used `a48ca71` plus the canary change
from `53cd403`, which was applied before the full capture. The reference's canary entries were then
regenerated with `regen_canaries.py`, and the anchor entries are as first captured. The capture host
had no git, so the manifest doesn't record the commit. Its report was re-scored after the p90 gate
change, with the same states and values.

The targeted re-run is in `tasks/benchmark-design/artifacts/mi355x-fill/` (106 MB). It holds the plan,
the full run's reference, `llmd --help` from the image, and the scripts that planned it
(`plan_fill.py`), rehearsed it against the fake server (`rehearse_fill.py`) and ran it. It used
`ab4ab8f`. The p90 gate change after it doesn't alter its scoring, because `fill` has no p99.

**Stand-ins:** the organizer's runner, schemas, fixtures and token adapter don't exist in this draft.
`collect/capture.py` and `schemas/` are our own, and `evaluate.py` depends only on the event format, so
adapting to the released format should mainly touch `analysis/events.py`.

**Integrity:** `python -B -m tools.check_integrity --base 81aead8` passes: protected organizer files are
unchanged from the starter commit `81aead87d78ed468c4c374fb42921471b44c3f80`. The `takehome-start` tag
the instructions name doesn't exist in this draft, so the check is given the commit directly. The
organizer's 40 root tests pass with git tag signing disabled for that one command. With the
contributor's `tag.gpgsign=true`, 15 of them error, and that also happens on a clean export. The
enhanced Pi profile wasn't supplied and wasn't used.

## Effort and resource use

| Resource | Use |
| --- | --- |
| Allocated GPU-hours | 0 of 12 |
| Local RTX 3060 | About 1 h of debug runs, not part of the allowance |
| Own MI355X, Daytona on-demand | 2.86 sandbox-hours: the full run with setup and the canary fix (2.23 h), a quick run stopped after 10 minutes, and the targeted re-run (0.46 h) |
| Active hours | 10 |
| API charges | $25, for the Daytona MI355X time above |

Cuts: checks with no named failure point were removed, namely stream ordering, model ID matching, a
token-rate ceiling, and output-difference reporting. Malformed streams now count as ordinary request
failures.
