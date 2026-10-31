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
| `chat_light` | 256 tokens | Up to 256 | 25% of capacity | Baseline TTFT and smoothness with no queueing |
| `long_prompt` | 4096 tokens | Up to 256 | 25% of capacity | Delayed prefill, which fast decode averages would hide |
| `long_output` | 128 tokens | Exactly 1024, `ignore_eos` | 25% of capacity | Decode stalls and use of slack |
| `near_capacity` | 80% short, 20% long | Up to 256 | 75% of capacity | Queueing, drops and unfair scheduling |
| `prefix_hit` | 4096 tokens, shared prefix | Up to 256 | 25% of capacity | Prefix-cache behavior |

Arrivals are seeded Poisson, because bursts are what real clients send, and a closed loop would hide
queueing by slowing down with the server. Rates are set as a share of measured capacity, so they
remain meaningful on any hardware. `long_output` uses `ignore_eos` so that decode length doesn't
depend on when the model chooses to stop.

Omitted: multi-turn conversations, tool calls, multi-GPU serving, and prompts over 4096 tokens. Prompts
are built from this repository's documents and code, so they're less realistic than real chat
traffic. `cold_fs` needs host rights to drop the page cache, so results report `warm_fs` only.

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

Real LLMD changed four rules that the fake server had let pass:

| Original rule | What real LLMD did | Rule now |
| --- | --- | --- |
| Canary text equals the answer | 0 of 120 passed, because the model shows its working | The last number must match; canaries the model gets wrong in the reference run are never sent |
| Usage equals the streamed token count | `usage` includes `<think>` tags that are never streamed | Hidden tags are added to the count |
| Usage within 1 token | Retokenizing fell short by up to 3 of 256 tokens | Tolerance is the larger of 1 token or 2% |
| Anchor text equals the reference | Fresh sequential: 24 of 24 matched. With or after concurrent traffic: 7 to 19 of 24 | First 8 characters, finish reason and at least half the reference length |

Because the anchor rule is weaker, a server returning plausible wrong text with the right opening and
length passes anchors. Canaries are the remaining defense, and they only show that the server
answers arithmetic correctly under load. Reasoning output often opens the same way, so the
8-character check mostly catches placeholders. A server that generates everything, then releases
tokens at a steady pace, can't be told apart from true streaming, though its cost appears in TTFT.

Fail-fast stops capture only when a run is already certain to be `invalid`. Healthy runs always finish,
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
| Cells | 2 | 5 |
| Repetitions per cell | 5 | 8 |
| Window | 120 s | 180 s |
| Launches | 3 | 5 |
| Quantiles | p50, p90 | p50, p90, p99 |

**Quantiles** are reported only when `n(1-q) >= 5`, so p99 needs 500 requests. Below that, it's close
to the maximum, and the result is `quantile_underpowered`.

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

Identical runs show that the unpaired interval is too wide. Two A/A runs differed by only 0.002 and
0.003 log on TTFT, yet both came back `inconclusive`, because the interval reflects different schedules
across repetitions, not disagreement between runs. Comparing matching repetitions as pairs should
tighten this, but it needs real A/A data before being adopted.

**Pending:** margins, deadlines and `max_anchor_divergence` from MI355X A/A runs, and the paired test.

## Validation matrix

All results below are debug or synthetic evidence, never MI355X validation.

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
run. `near_capacity` ran close to overload here, so its wide intervals say little about the design.

| Run | State | Failed | Anchors diverged | TTFT p90 `chat_light` | Against `clean_a` |
| --- | --- | --- | --- | --- | --- |
| `clean_a` | `valid` | 0 | 0 of 41 | 42 ms | Baseline |
| `clean_b` | `valid` | 0 | 0 of 41 | 42 ms | Fluidity `equivalent`; TTFT and startup `inconclusive`; no false regression |
| `--batch-size=1` | `valid` | 0 | 0 of 41 | 59.7 s | TTFT `regression` on both cells; fluidity `regression` on `chat_light`, `inconclusive` on `near_capacity`; startup `inconclusive` |

The degraded run is a real engine setting, not an injected fault. Startup took about 6 s and wasn't
expected to change.

**Fake server over real HTTP:** a clean run, wrong answers and a crash were re-run on the final code.
Wrong answers stopped at 48 s and a crash at 13 s, both `invalid`. False usage and slow TTFT last ran
before the out-of-scope checks were removed, when they were caught as `invalid` and `regression`.

**Limitations:** only one real degradation has been tested. Real faults on the MI355X will be limited
to what can be produced without changing the engine, so most fault coverage stays synthetic. The real
A/A pair is a single sample, not a false-positive rate. Fake-server results are never validation
evidence.

**Pending:** A/A controls and real slowdowns on the MI355X.

## Measured compute cost

Measured on the debug plan only. This says nothing about the target, whose load and compile time could
be far longer.

| Run | Wall time | Launch | Warmup | Requests | Drain | Cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| `clean_a` | 403 s | 17 s | 18 s | 357 s | 8 s | 2 s |
| `clean_b` | 403 s | 18 s | 18 s | 357 s | 8 s | 2 s |
| `--batch-size=1` | 759 s | 17 s | 17 s | 357 s | 366 s | 2 s |

The degraded run shows why drain is measured: overload doubled the run's cost, all of it after the
request windows ended.

Planned request time, excluding every other phase:

| Mode | Request time | Wall limit in plan |
| --- | --- | --- |
| Quick | 2 × 5 × 120 s = 20 min | 40 min |
| Full | 5 × 8 × 180 s = 2 h | 2.5 h, inside the 3 h ceiling |

**Pending:** measured quick and full cost on the MI355X. That the full plan fits is a calculation
until then, not a result. No GPU-hours have been used against the 12-hour allowance.

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
| `collect/tokenize_events.py` | Independent token counts and times |
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
# Real LLMD on a CUDA GPU, from suite/, with requirements.txt installed
python tests/e2e/e2e_cuda.py --out <dir> --model-dir <Qwen3-0.6B directory>
```

| Identity | Value |
| --- | --- |
| `workload.json` SHA-256 | `175f2c447a7e63c5163f2607f373f541fca2f08acf5b27d437beb0c8e085e98e` |
| Seed | 20261030 |
| Debug image | `zmlai/llmd@sha256:752fc2a8a50971a14d2035e653c6c8ddf672d257b3cfed7eb43f216a123bda7c` |
| Debug model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Prompt tokenizer | `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218` |

Raw debug evidence is in `tasks/benchmark-design/artifacts/debug-cuda/` (247 MB). The path is ignored
by git, so it needs separate delivery. Its reports were re-scored after the last code cut.

**Stand-ins:** the organizer's runner, schemas, fixtures and token adapter don't exist in this draft.
`collect/capture.py` and `schemas/` are our own, and `evaluate.py` depends only on the event format, so
adapting to the released format should mainly touch `analysis/events.py`.

**Integrity:** protected paths show no diff from `81aead8`. `tools.check_integrity --base takehome-start`
hasn't produced a passing result, because the `takehome-start` tag doesn't exist in this draft. The
organizer's 40 root tests pass with git tag signing disabled for that one command. With the
contributor's `tag.gpgsign=true`, 15 of them error, and that also happens on a clean export. The
enhanced Pi profile wasn't supplied and wasn't used.

## Effort and resource use

| Resource | Use |
| --- | --- |
| Allocated GPU-hours | 0 of 12 |
| Local RTX 3060 | About 1 h of debug runs, not part of the allowance |
| Active hours | To be filled in by the candidate |
| API charges | To be filled in by the candidate |

Cuts: checks with no named failure point were removed, namely stream ordering, model ID matching, a
token-rate ceiling, and output-difference reporting. Malformed streams now count as ordinary request
failures.
