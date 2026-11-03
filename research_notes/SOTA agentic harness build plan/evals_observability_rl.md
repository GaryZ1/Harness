# Evaluation, Observability, Production Operations, and RL/Training Co-Design for Agentic Harnesses (state as of 2026-11-03)

Research note on method and access: Several primary-source hosts could not be reached from this session: metr.org, tbench.ai, arxiv.org, opentelemetry.io, huggingface.co, platform.openai.com, and the GitHub API for laude-institute/harbor. anthropic.com could be fetched directly. Everything else comes from web-search result summaries, which mix primary pages with aggregators. Each finding names its source type. Leaderboard aggregators (BenchLM, llm-stats, llmlearner, Snorkel, Vals.ai) often disagree with each other, so treat any single number as "reported by X on date Y", not as ground truth.

## Benchmarks: current SOTA, the harnesses used, and known issues

### Takeaway
Every 2025-era flagship agent benchmark is now saturated, contaminated, or audited as partly broken:
- SWE-bench Verified: ~97% with a bash-only harness.
- OSWorld-Verified: ~86-90%, against a human baseline of 72%.
- tau2 telecom: ~99%.
- SWE-bench Pro: the same model scores ~59-61% on Scale's standardized harness but ~90% in vendor-reported numbers.

Frontier reporting has moved to harder or newer suites (SWE-bench Pro, Terminal-Bench 2.x and reportedly 4.0, OSWorld 2.0, GDPval, METR TH1.1). Scores swing by several to more than 20 points depending on the harness, so every number has to be read together with its harness, effort setting, and run count.

### Cited Findings

**SWE-bench Verified (500 tasks, Python repos)**
- As of 2026-09-01, Claude Opus 5 with mini-SWE-agent ranked first on SWE-bench Verified at 97.00% (±0.76). Next were DeepSeek V4 Pro 0813 at 96.40% and GPT-5.6 Sol at 96.20%, all on the mini-SWE-agent bash-only harness, run independently by Vals.ai in isolated Docker containers. The Vals page is marked archived. — [Vals.ai SWE-bench page](https://www.vals.ai/benchmarks/swebench) (via search summary)
- Harness definition: "bash-only" gives the model one tool, bash. It must navigate, search, edit and solve using standard CLI tools. The SWE-bench project now treats mini-SWE-agent runs as regular Verified submissions; "Bash Only" is a filter on the Verified board. — [swebench.com Verified](https://www.swebench.com/verified.html); [mini-swe-agent repo](https://github.com/swe-agent/mini-swe-agent) (the README says ">74% on SWE-bench verified" with a ~100-line agent)
- 2026-02-23: OpenAI announced that it no longer reports SWE-bench Verified and recommended SWE-bench Pro instead. Its reasons were saturation from test-design issues and contamination from public repos. Every frontier model it tested could reproduce some gold patches. Problems were found in 59.4% of the 138 *failed* tasks it audited, which is not 59.4% of the whole benchmark. — [OpenAI Devs on X](https://x.com/OpenAIDevs/status/2026002219909427270); summarized by [codesota](https://www.codesota.com/news/swe-bench-contamination-debate) and [byteiota](https://byteiota.com/openai-abandons-swe-bench-verified-59-flawed-tests/)
- Epoch AI rates SWE-bench Verified as "Flawed". — [secondary commentary citing Epoch](https://www.arturmarkus.com/ai-model-comparisons-5-myths-from-the-benchmark-openai-retired-to-the-0-08-model-that-scores-21/); [Epoch SWE-bench Verified page](https://epoch.ai/benchmarks/swe-bench-verified)
- SWE-ABS (2026) uses adversarially strengthened, augmented test suites. Under these tests, bash-only agent scores drop substantially, which suggests that test-based resolve rates are inflated. — [SWE-ABS arXiv 2603.00520](https://arxiv.org/pdf/2603.00520)
- Anthropic's own eval guide notes that on SWE-bench Verified, LLMs went "from 40% to >80%" in about a year. — [Anthropic, Demystifying evals for AI agents, 2026-01-09](https://anthropic.com/engineering/demystifying-evals-for-ai-agents)
- SWE-bench Feb 2026 leaderboard update. — [Simon Willison, 2026-02-19](https://simonwillison.net/2026/Feb/19/swe-bench/) (title only seen)

**SWE-bench Pro (Scale AI; 731-task public split)**
- On Scale's standardized public set in October 2026, Muse Spark 1.1 led at 61.5% and GPT-5.4 (xHigh) was second at 59.1%. — [Morph SWE-bench Pro page, Oct 2026](https://www.morphllm.com/swe-bench-pro)
- Vendor-reported/aggregated: Claude Opus 5.5 at 89.9%, then Claude Sonnet 5.5 at 81.3% and Claude Fable 5.1 at 81.2% (BenchLM update 2026-10-29). llm-stats also lists Opus 5.5 first at 0.899. — [BenchLM SWE-bench Pro](https://benchlm.ai/benchmarks/swe-bench-pro); [llm-stats](https://llm-stats.com/benchmarks/swe-bench-pro)
- The Opus 5.5 announcement and system card reportedly do *not* report SWE-bench Verified. They report SWE-bench Pro 89.9%, SWE-bench Multilingual 93.9%, and SWE-bench Multimodal 61.4%. — [Vellum, Opus 5.5 benchmarks explained](https://www.vellum.ai/blog/claude-opus-5-5-benchmarks-explained); [llm-stats Opus 5.5 launch](https://llm-stats.com/blog/research/claude-opus-5-5-launch) (secondary; I could not reach the Anthropic primary page)
- Reported but **unverified**: on 2026-07-08 OpenAI classed ~30% of SWE-bench Pro's 731 public tasks as broken and withdrew its interim recommendation to adopt Pro. — reported by [BenchLM](https://benchlm.ai/benchmarks/swe-bench-pro) and [codesota](https://www.codesota.com/news/swe-bench-contamination-debate); not confirmed against a second source
- Harness gap on the same model (secondary): Scale's standardized SEAL scaffold put Claude Opus 4.5 at 45.9% on SWE-bench Pro, versus 55.4% for the same model inside Claude Code. — [particula.tech](https://particula.tech/blog/agent-scaffolding-beats-model-upgrades-swe-bench) (unverified against Scale)
- SWE-Bench Pro "V2" on Fireworks shows Claude Opus 5 at 99.38% and Kimi K3 (best open model) at 97.7%. This is a different benchmark version and not comparable. — [Fireworks](https://fireworks.ai/specialized-intelligence-index/benchmarks/code-generation/swe-bench-pro-v2)
- Augment's Auggie claimed the top of SWE-Bench Pro at some point. — [Augment blog](https://www.augmentcode.com/blog/auggie-tops-swe-bench-pro) (date not captured)

**Terminal-Bench 2.0 / 2.1 (and Harbor)**
- TB2 paper (arXiv 2601.11868, Jan 2026): 89 tasks, each manually verified by three human reviewers for specificity, solvability and integrity. They were selected from ~229 crowdsourced tasks by 93 contributors (Snorkel says "nearly 100 developers"). Unreproducible or arbitrarily threshold-gated tasks were removed. At release, frontier agents solved <65% and small models ~15%. Terminus 2 is the neutral baseline harness. — [TB2 paper via hyper.ai](https://hyper.ai/en/papers/2601.11868); [arXiv 2601.11868](https://arxiv.org/html/2601.11868v1); [Snorkel](https://snorkel.ai/leaderboard/terminal-bench-2-0/)
- Terminal-Bench v2.1 is "a verified refresh of v2.0 — 89 curated tasks, with environment and instruction fixes". — [Artificial Analysis](https://artificialanalysis.ai/evaluations/terminalbench-v2-1)
- Leaderboard submissions require five benchmark runs, and you run them through the Harbor CLI. — [VentureBeat on TB2 + Harbor launch](https://venturebeat.com/ai/terminal-bench-2-0-launches-alongside-harbor-a-new-framework-for-testing)
- Model-level, late Oct 2026: GPT-5.5 82% (BenchLM) or 0.827 (llm-stats), then GPT-5.3 Codex 77.3% and GPT-5.4 75.1%. llmlearner lists Claude Mythos Preview at 82.0. — [BenchLM TB2](https://benchlm.ai/benchmarks/terminal-bench-2); [llm-stats TB2](https://llm-stats.com/benchmarks/terminal-bench-2); [llmlearner](https://llmlearner.com/rankings/terminalbench-2)
- Harness-level, Snorkel: NexAU-AHE on GPT-5.5 at 84.7% ±2.1, LemonHarness 84.5%, Capy on GPT-5.5 83.1%, WOZCODE on Claude Opus 4.7 80.2%. Harness-level results run higher than model-level ones. — [Snorkel TB2 leaderboard](https://snorkel.ai/leaderboard/terminal-bench-2-0/)
- Same model, different wrapper (secondary): GPT-5.2-Codex scored 64.7% in the Codex CLI wrapper versus 57.5% in Terminus-2, a 7.2-pt gap. — [harness-sensitivity search summary; dev.to / particula blogs](https://particula.tech/blog/agent-scaffolding-beats-model-upgrades-swe-bench)
- **Terminal-Bench 4.0** (reported, not verified against tbench.ai): Anthropic self-reports Opus 5.5 at 66.4% (±2.6) at xhigh effort with production safeguards. Artificial Analysis measured 59.6%. Vals.ai measured 65.15%, falling to 53.54% when fallback-assisted tasks count as failures. — [Vellum](https://www.vellum.ai/blog/claude-opus-5-5-benchmarks-explained); [Vals.ai Opus 5.5](https://www.vals.ai/models/anthropic_claude-opus-5-5)
- Reportedly, the default API effort dropped from "high" on Opus 5 to "medium" on Opus 5.5. That is a harness setting that changes scores. — [Vellum](https://www.vellum.ai/blog/claude-opus-5-5-benchmarks-explained) (secondary)

**tau-bench / tau2-bench (Sierra)**
- pass^k = probability that the agent succeeds on *all* k independent trials of the same task. GPT-4o's tau-retail agent fell to ~25% at pass^8, about 60% below its pass^1. — [Sierra, Benchmarking AI agents](https://sierra.ai/uk/blog/benchmarking-ai-agents) (2024 background)
- tau2-bench extends tau-bench with code fixes and a telecom troubleshooting domain where the *user also has tools* and the agent must guide them. — [evalscope tau2 docs](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html)
- Telecom is saturated. BenchLM on 2026-10-25 shows GLM-5.2 at 99.1% and GPT-5.4 at 98.9%, mostly from Artificial Analysis's telecom implementation. — [BenchLM tau2](https://benchlm.ai/benchmarks/tau2-bench)
- Airline: OpenRouter's live board (updated 2026-11-01; averages the newest 90 days of runs) shows GPT-5.1 leading at 80.0%. — [OpenRouter tau2 airline](https://openrouter.ai/benchmarks/tau2-bench-airline)
- The user-simulator and judge configuration matters. One listing gives 70.2% "measured using the 'High' user simulator and GPT-5.2 judge protocol." — [steel.dev tau-bench](https://leaderboard.steel.dev/leaderboards/tau-bench/)
- Changing only the scaffold flag on tau-bench moves the same model by up to 0.267 and can flip model rankings. — [arXiv 2609.09218, "Double Measurement Confound"](https://arxiv.org/pdf/2609.09218)

**OSWorld / OSWorld-Verified / OSWorld 2.0**
- Human baseline on OSWorld-Verified is 72.36%. As of 2026-10-01/13, the top entry is Intelligence-Indeed Agent at 90.19% (dated 2026-07-25), then Claude Fable 5 at 85.96%. 16 of 149 verified entries beat the human baseline. — [presenc.ai](https://presenc.ai/research/osworld-computer-use-leaderboard-2026); [decodethefuture](https://decodethefuture.org/en/ai-agent-benchmarks-2026/)
- llm-stats (Sept 2026) instead shows Qwen3.8 Max leading at 0.861 across 24 models. Leaders vary by source. — [llm-stats OSWorld-Verified](https://llm-stats.com/benchmarks/osworld-verified)
- Earlier snapshot (2026-05-26): Pointer claimed 83.6% with Claude Opus 4.7 and 81.5% with Sonnet 4.6. — [Pointer blog](https://www.pointer.ai/blog/sota)
- **OSWorld 2.0**: Opus 5.5 is announced at 81.8% *with partial credit*, but its strict completion rate is 48.7% per the system card (as summarized). Partial credit versus strict scoring is a major reporting trap. — [llm-stats Opus 5.5 launch](https://llm-stats.com/blog/research/claude-opus-5-5-launch)

**BrowseComp, GDPval, HLE with tools**
- GPT-5.5 announcement (self-reported, ~6 months old): GDPval 84.9%, BrowseComp 84.4%, and GPT-5.5 Pro BrowseComp 90.1%. — [OpenAI, Introducing GPT-5.5](https://openai.com/index/introducing-gpt-5-5/)
- GDPval design: 44 occupations, 1,320 tasks, 9 U.S. sectors, and a 220-task open gold subset. Blinded expert pairwise comparison of the model deliverable against the human-expert deliverable gives win/tie/loss. Leaderboards report "win rate" and "wins+ties", so check which one a score uses. The automated grader agreed with human experts 66% of the time, against 71% human-human agreement. ICLR 2026. — [GDPval paper (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/290c2430f91912204f30bbcc990fff1d-Paper-Conference.pdf); [OpenAI GDPval post](https://openai.com/index/gdpval/); [Epoch GDPval](https://epoch.ai/benchmarks/gdpval)
- Anthropic's multi-agent research post found that token usage alone explained ~80% of BrowseComp performance variance. Tool-call count and model choice brought it to ~95%. — [Anthropic, multi-agent research system, 2025-06-13](https://www.anthropic.com/engineering/multi-agent-research-system)
- HLE with tools (Oct 2026): Claude Opus 5.5 at 67.7%, Sonnet 5.5 at 64.5%, Muse Spark 1.1 at 62.1% (BenchLM). llm-stats shows Sonnet 5.5 leading at 0.645 across only 5 models. The HLE-Diamond variant with web and code tools has GPT-6 Astra at 82.9%. The text-only Scale board on 2026-10-29 has GPT-6 Astra at 54.8% and Claude Fable 5.1 at 46.5%. — [BenchLM HLE](https://benchlm.ai/benchmarks/scale-humanitys-last-exam); [llm-stats HLE w/ tools](https://llm-stats.com/benchmarks/humanity's-last-exam-(with-tools)); [benchleader HLE](https://www.benchleader.com/benchmarks/hle)

**METR time horizons**
- Background (March 2025): the 50% time horizon doubled roughly every 7 months over 2019-2024. — [METR, arXiv 2503.14499](https://arxiv.org/pdf/2503.14499)
- Time Horizon 1.1 (2026-01-29) grew the suite from 170 to 228 tasks and the 8h+ tasks from 14 to 31, and re-estimated 14 models. The post-2023 doubling time is 131 days under TH1.1, against 165 days under TH1. For 2024 onward, the P50 doubling time is 88.6 days. — [METR TH1.1 blog](https://metr.org/blog/2026-01-29-time-horizon-1-1/) (via search snippets)
- Scaffolds: most METR measurements use **Triframe and ReAct**. Results are "somewhat sensitive to scaffold (including prompting)". In one case the older Vivaria scaffold elicited slightly stronger performance. METR also measured time horizon *using Claude Code and Codex* as the harness (note dated 2026-02-13). — [METR TH1.1](https://metr.org/blog/2026-01-29-time-horizon-1-1/); [METR note: Measuring Time Horizon using Claude Code and Codex](https://metr.org/notes/2026-02-13-measuring-time-horizon-using-claude-code-and-codex/)
- Claude Mythos Preview: 50% horizon at least 16h (95% CI 8.5h-55h) and 80% horizon ~3h06m. METR says "measurements above 16 [hours] are unreliable with [their] current task suite", and TH1.1 is close to saturated. — [METR via search summary](https://metr.org/blog/2026-01-29-time-horizon-1-1/); [EA Forum, TH2.0 discussion](https://forum.effectivealtruism.org/posts/ghhJh6SBYXxDktnZX/metr-time-horizon-2-0-the-benchmark-you-ve-been-waiting-for)
- A 105-day doubling (through Feb 2026 data) is a secondary claim. — [X post](https://x.com/aakashgupta/status/2053207639107174452) (unverified). The AI Futures Project shortened its assumed doubling time to 4-4.5 months (2026-04-02). — [AI 2027 tracker](https://ai2027-tracker.com/predictions/metr-doubling/)
- Critiques of estimation validity. — [arXiv 2610.12466](https://arxiv.org/pdf/2610.12466v1)

**MLE-bench / PaperBench**
- No authoritative current leaderboard was found. Aggregator data: MLE-bench full (75 tasks) top is Gemini 3.6 Flash at 63.9 (vendor-reported, July 2026; history from 42.6 in Feb 2026). PaperBench top is Qwen3.8 Max at 93.0% (BenchLM, Aug 2026, likely stale). ML-Master 2.0 with DeepSeek-V3.2-Speciale reports 75.8 "Any Medal" on MLE-bench Lite. — [llmlearner MLE-bench](https://llmlearner.com/rankings/mle-bench); [BenchLM PaperBench](https://benchlm.ai/benchmarks/paperbench); [BenchLM MLE-bench Lite](https://benchlm.ai/benchmarks/mle-bench-lite)

**Harness sensitivity and run-to-run noise (cross-cutting)**
- Re-running the same model and scaffold over 60,000 SWE-bench trajectories moves single-run pass@1 by 2.2-6.0 pts depending on which run you pick. — [dev.to summary of arXiv 2602.07150](https://dev.to/devinios/same-model-same-scaffold-10-runs-agent-benchmark-scores-still-move-by-up-to-6-points-gmb)
- Claude-4.5-Opus gains 20.5% inside the Claude Code SDK compared with a generalist scaffold. — [AgencyBench arXiv 2601.11044](https://arxiv.org/pdf/2601.11044)
- On GAIA, single-scaffold numbers are "scaffold-conditional point estimates, not model properties", and scaffold sensitivity does not shrink as capability grows. — [arXiv 2606.08529](https://arxiv.org/pdf/2606.08529)
- "We improved 15 LLMs at coding in one afternoon. Only the harness changed" (2026-02-12). — [Stencil blog](https://blog.can.ac/2026/02/12/the-harness-problem)
- Leaderboard rank comparability is examined in [arXiv 2609.07785](https://arxiv.org/pdf/2609.07785); infrastructure argument in the Holistic Agent Leaderboard (HAL). — [arXiv 2510.11977](https://arxiv.org/pdf/2510.11977)

### Inferences
- For a harness builder, public benchmarks are useful as **harness A/B tests with the model held fixed**, not as model rankings. Use Terminal-Bench 2.x via Harbor, SWE-bench Pro, and tau2 airline/retail with pass^k, and always report: harness version, effort/thinking setting, number of runs (TB2 requires 5), mean ± SE, partial-credit versus strict scoring, and any "fallback-assisted" counting.
- SWE-bench Verified (~97%) and tau2-telecom (~99%) work only as *regression* smoke tests now. They no longer measure capability.
- Since even "verified" suites carry 30-60% broken-task rates among failures, any internal eval failures above ~50% pass rate need transcript audits before you trust them.
- The 89-task TB2 size means ±2-3 pts SE per model, so a harness change needs roughly >5 pts to be distinguishable from noise without many repeats.
- Recall from the DeepSeek-V3.2 release (unverified in this session; [paper](https://arxiv.org/pdf/2512.02556)): SWE-bench Verified ~73.1, Terminal Bench 2.0 ~46.4, BrowseComp ~51.4 (rising to ~67.6 with context-management strategies), tau2 ~80.3. These numbers are useful as an open-weights reference point from late 2025.

### Gaps
- Could not reach tbench.ai, swebench.com, os-world.github.io, taubench.com, metr.org, or Epoch AI directly. Official current leaderboard rows (especially TB2 official top 10 with dates, and SWE-bench Multimodal SOTA beyond Opus 5.5's reported 61.4%) are unverified.
- Terminal-Bench 4.0 and OSWorld 2.0 appear only in secondary write-ups about Opus 5.5. Their task counts, release dates and maintainers are unconfirmed.
- GAIA 2026 SOTA was not found. The "HAL scaffold adds ~30 pts on GAIA" claim is secondary and unverified.
- No current official BrowseComp leaderboard was found. Only self-reported numbers in launch posts.
- Models such as "Claude Fable 5.x", "GPT-6 Astra", "Muse Spark 1.1", and "Qwen3.8 Max" appear only on aggregators. I did not verify their launch dates.

## Eval methodology for builders (graders, pass@k vs pass^k, suites, transcripts)

### Takeaway
The consensus methodology (Anthropic, Jan 2026; OpenAI's trace-grading docs; Sierra) is:
- Start with 20-50 tasks drawn from real failures.
- Grade *end-state outcomes* with deterministic checks first. Add LLM rubrics calibrated against humans for the rest.
- Run multiple isolated trials and report pass@k or pass^k depending on the product.
- Split suites into capability suites (low pass rate, hill-climb) and regression suites (~100%).
- Read transcripts weekly so you are validating the graders as well as the agent.

### Cited Findings
**Anthropic, "Demystifying evals for AI agents" (2026-01-09)** — [source](https://anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Vocabulary:
  - **task**: inputs plus success criteria.
  - **trial**: one attempt.
  - **grader**: scoring logic with assertions; there can be several per task.
  - **transcript**: the full record, including tool calls and reasoning.
  - **outcome**: the final environment state, e.g. whether the reservation exists in the DB, not whether the agent *said* it booked.
  - **evaluation harness**: runs tasks concurrently, records, grades, aggregates.
  - **agent harness/scaffold**: "When we evaluate 'an agent,' we're evaluating the harness *and* the model working together."
- Grader types:
  - **Code-based**: string/regex, fail-to-pass/pass-to-pass tests, static analysis, outcome and tool-call verification, transcript metrics. Fast, cheap, reproducible; brittle.
  - **Model-based**: rubric, NL assertions, pairwise, reference-based, multi-judge consensus. Flexible; nondeterministic; needs calibration.
  - **Human**: SME review, spot checks, A/B tests, inter-annotator agreement. The gold standard; used to calibrate model graders.
- Scoring modes: weighted (threshold), binary (all graders must pass), or hybrid.
- **pass@k** = P(≥1 success in k). **pass^k** = P(all k succeed) = p^k. Example: 0.75³ ≈ 42%. Use pass^k for customer-facing agents where consistency matters.
- **Capability evals** start at a low pass rate ("a hill to climb"). **Regression evals** sit near 100%. Capability tasks "graduate" into regression suites.
- Coding-agent example task YAML combines unit tests, an `llm_rubric`, static analysis (ruff, mypy, bandit), a `state_check`, `tool_calls` requirements, and tracked metrics (turns, tool calls, tokens, latency). In practice teams start with unit tests plus an LLM rubric.
- Conversational agents need an LLM-simulated user. Success is multi-dimensional: state check plus transcript constraint (e.g., `max_turns: 10`) plus rubric (tone, grounding); a tool constraint example is refund `amount: "<=100"`.
- Research agents: groundedness, coverage (required key facts), and source quality; exact match where the answer is objective; recalibrate rubrics against experts often.
- Computer-use agents: verify backend, file system, DB, URL or page state in a sandbox. DOM is fast but token-heavy; screenshots are slower but token-efficient.
- Roadmap:
  1. "20-50 simple tasks drawn from real failures is a great start." Early effect sizes are large, so small samples suffice.
  2. Source tasks from manual pre-release checks, bug trackers, and support queues.
  3. Write unambiguous tasks with a **reference solution**. Test: two experts would agree. "0% pass@100 is most often a signal of a broken task."
  4. Balance positive and negative cases (e.g., web search under- versus over-triggering).
  5. **Isolate trials**: clean environment per trial. Claude once read git history from previous trials, and shared memory limits make trials non-independent.
  6. Grade outcome, not path (rigid step checks are "overly brittle"). Use partial credit. Give LLM judges an "Unknown" option. Use **one isolated judge per dimension**. Make graders hack-resistant.
  7. Case study: Opus 4.5 scored 42% on CORE-Bench until grader bugs were fixed (e.g., rejecting "96.12" when "96.124991…" was expected). It then reached 95% with a less constrained scaffold.
  8. Read transcripts; "failures should seem fair."
  9. Monitor saturation: "An eval at 100% tracks regressions but provides no signal for improvement."
  10. Eval-driven development: write failing evals before building a capability. Non-engineers can contribute tasks as PRs via Claude Code.
- "Swiss cheese" layers: automated evals (CI/CD, every change and model upgrade), production monitoring, A/B tests, user feedback (sparse, skews severe), weekly transcript sampling, systematic human studies to calibrate graders.
- Frameworks named: Harbor (containerized runs, standard task format, TB2 registry), Braintrust (`autoevals`), LangSmith, Langfuse (self-hosted), Arize Phoenix/AX. Advice: "pick a framework quickly, then invest in high-quality tasks and graders."

**Anthropic multi-agent research system (2025-06-13)** — [source](https://www.anthropic.com/engineering/multi-agent-research-system)
- Started with about 20 queries from real usage. One prompt tweak moved success from 30% to 80%.
- LLM-as-judge rubric dimensions: factual accuracy, citation accuracy, completeness, source quality, tool efficiency. A *single LLM call* producing 0.0-1.0 scores plus pass/fail was the most consistent and closest to human judgment.
- Human testers caught that agents preferred SEO content farms over authoritative sources. Fixed with source-quality heuristics in the prompt.

**Anthropic tool evals (2025-09-11)** — [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Have Claude Code generate dozens of realistic prompt/response tasks that need many tool calls, each with a verifiable outcome (exact match or Claude-as-judge, tolerant of format variations).
- Run each task through a simple agentic loop and ask the agent to emit reasoning and feedback before tool calls.
- Collect accuracy, tool and task runtime, number of tool calls, token consumption, and tool errors.
- Use a held-out test set to avoid overfitting.

**OpenAI evals / trace grading**
- Trace grading means assigning structured scores or labels to an agent's end-to-end trace (decisions, tool calls, reasoning). The dashboard flow is Logs > Traces → pick a workflow → create a grader → run. **Trace evals** reuse graded traces across many examples for benchmarking and regression. — [OpenAI Trace grading guide](https://developers.openai.com/api/docs/guides/trace-grading)
- "Evaluate agent workflows": start with trace grading while debugging, and move to dataset graders once behavior stabilizes. "Trace grading is the fastest way to identify workflow-level issues." — [OpenAI agent-evals guide](https://developers.openai.com/api/docs/guides/agent-evals)
- AgentKit (Oct 2025) launched new Evals features (datasets, trace grading, automated prompt optimization, third-party model support) alongside RFT for agents. — [OpenAI, Introducing AgentKit](https://openai.com/index/introducing-agentkit/)
- Third-party: `agentevals-cli` scores existing OTel traces without re-running LLM calls and can delegate grading to the OpenAI Evals API. — [PyPI agentevals-cli](https://pypi.org/project/agentevals-cli/)

**Reliability metric (Sierra)**: pass^k was introduced with tau-bench. GPT-4o fell from pass^1 to ~25% pass^8 on retail. — [Sierra blog](https://sierra.ai/uk/blog/benchmarking-ai-agents)

**LLM-judge reliability datapoint**: on GDPval, the automated grader agreed with experts 66% of the time against 71% human-human agreement. Judges get close to, but stay below, inter-expert agreement on open-ended professional work. — [GDPval paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/290c2430f91912204f30bbcc990fff1d-Paper-Conference.pdf)

### Inferences
- Concrete eval stack for a harness project:
  - **Tier 0**: tool-level unit evals per tool (schema adherence, error messages, token budget).
  - **Tier 1**: ~50-task capability suite from real failures with reference solutions, outcome graders and isolated containers (Harbor task format is a reasonable standard).
  - **Tier 2**: regression suite of graduated tasks run on every PR or model bump.
  - **Tier 3**: public suites (TB2.x via Harbor, SWE-bench Pro, tau2 airline/retail) for periodic external calibration.
  - **Tier 4**: online trace grading on sampled production traffic.
- Report pass@1 as the mean over ≥3-5 trials, plus pass^k (k=3-8) for user-facing reliability. A single run carries a 2-6-pt noise band.
- Track efficiency metrics (turns, tool calls, tokens, cost, wall-clock) as first-class eval outputs, not just pass/fail. Anthropic's YAML and OpenAI's RFT results both treat fewer turns and more parallel calls as wins.
- Budget for grader QA. CORE-Bench (42% → 95%) and SWE-bench (59% of audited failures flawed) show grader bugs can dominate measured failure.

### Gaps
- Could not fetch OpenAI's agent-evals and trace-grading docs in full (platform.openai.com blocked; developers.openai.com only via search). Exact grader types (string_check, text_similarity, score_model, label_model, python, multi) are not confirmed in this session.
- No primary source found for "tool-level eval" frameworks beyond Anthropic's tool-writing post.

## Observability (OTel GenAI conventions, SDK tracing, Claude Code telemetry, vendor platforms)

### Takeaway
OpenTelemetry GenAI semantic conventions define a standard span tree, `invoke_agent` → `chat` (one per model call) → `execute_tool`, with token-usage attributes. They are still in *Development* status and moved to a dedicated repo in mid-2026. Instrument against them so traces are portable across Langfuse, Phoenix, Braintrust and LangSmith. Manus, Codex and Anthropic all treat **prompt-cache hit rate** as a top-line production metric alongside cost, latency and tool error rate.

### Cited Findings
- Agent spans: `gen_ai.operation.name` SHOULD be `invoke_agent`, with `create_agent` for creation and `execute_tool` for tool execution. The span name is `invoke_agent {gen_ai.agent.name}` when the name is available, else `invoke_agent`. Span kind is CLIENT (INTERNAL allowed for in-process agents). — [OTel GenAI agent spans spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans) (via search snippet; site not fetchable)
- Typical tree: parent `invoke_agent` with `chat` and `execute_tool` children per loop step. Tool spans carry tool name, call ID, arguments, result, and error status. — [DEV Community guide, 2026](https://dev.to/gabrielanhaia/opentelemetry-genai-semantic-conventions-your-llm-traces-should-look-like-this-in-2026-3ff6); [Morph agent tracing](https://www.morphllm.com/agent-tracing)
- Status and version conflict: the conventions are still "Development". One source says that on 2026-06-12, in v1.42.0, GenAI conventions moved into a dedicated `semantic-conventions-genai` repo. Another cites docs showing v1.41.0. — [Greptime blog 2026-05-09](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions); [Dash0 explainer, updated 2026-09-14](https://www.dash0.com/knowledge/opentelemetry-genai-semantic-conventions-explained)
- OpenAI Agents SDK: tracing is built in and on by default server-side. Each run records model calls, tool calls, handoffs, guardrails and custom spans, viewable in the Traces dashboard. A `withTrace` helper groups multiple runs into one trace. Trace processors can export to Braintrust and others. — [OpenAI Agents integrations & observability](https://developers.openai.com/api/docs/guides/agents/integrations-observability); [Braintrust Agents SDK integration](https://www.braintrust.dev/docs/integrations/openai-agents-sdk)
- Claude Code telemetry:
  - Enabled with `CLAUDE_CODE_ENABLE_TELEMETRY=1`. Exporters are `OTEL_METRICS_EXPORTER` and `OTEL_LOGS_EXPORTER` (`otlp`, `console`, `prometheus`; multiple values allowed). Endpoint and auth via `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS`.
  - Default metric export interval is 60,000 ms. Headers are read only at startup.
  - Metrics include `claude_code.cost.usage` and `claude_code.token.usage`, labeled by `token.type` (input/output/cacheRead/cacheCreation).
  - Events go via the logs exporter (e.g., `claude_code.api_request`). Prometheus conversion renames metrics (e.g., `claude_code_cost_usage_USD_total`).
  - Sources: [Claude Code monitoring docs](https://code.claude.com/docs/en/monitoring-usage?m=1); [Bindplane guide](https://docs.bindplane.com/how-to-guides/cloud-and-platform-integrations/send-claude-code-telemetry-to-bindplane); [li0nel/claude-otel](https://github.com/li0nel/claude-otel)
- Anthropic's production research agent used full production tracing to diagnose failures. It monitored "agent decision patterns and interaction structures" *without* reading conversation contents, for privacy. — [Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- Manus says "the KV-cache hit rate is the single most important metric for a production-stage AI agent" because it drives latency and cost. Average input:output ratio is ~100:1. Cache-breakers include timestamps at the start of the system prompt, non-deterministic serialization, and adding or removing tools mid-session (Manus masks logits instead). — [Manus, Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus?hl=en-GB)
- OpenAI "Unrolling the Codex agent loop" (Michael Bolin, 2026-01-23) describes the Codex harness on the Responses API. Secondary summaries say the old prompt must be an exact prefix of the new one for caching, dynamic tool lists (MCP) can break the prefix, and compaction rewrites history and so invalidates the cached prefix. — [OpenAI post](https://openai.com/index/unrolling-the-codex-agent-loop); [all-ai.de summary](https://www.all-ai.de/news/beitrage2026/codex-agent-loop); [danielvaughan Codex caching](https://codex.danielvaughan.com/2026/04/21/codex-cli-prompt-caching-maximise-cache-hits-cost-reduction/)
- Platform landscape (2026, secondary comparison blogs):
  - **Langfuse**: acquired by ClickHouse in Jan 2026; remains MIT and self-hostable; OTel-native.
  - **LangSmith**: online evals on live traffic and automatic trace clustering of failure modes; proprietary, with self-host on enterprise tier only.
  - **Braintrust**: eval-first, CI gates, "Loop" assistant, Topics clustering in beta.
  - **Arize Phoenix**: OTel-native, ELv2 source-available, free to self-host; AX is the SaaS.
  - Sources: [MarkTechPost 2026-08-09](https://www.marktechpost.com/2026/08/09/top-llm-observability-and-evaluation-platforms-in-2026-langfuse-langsmith-braintrust-arize-and-more-compared/); [latitude.so comparison](https://latitude.so/blog/best-ai-agent-observability-tools-2026-comparison); [laminar.sh](https://laminar.sh/article/langfuse-alternatives-2026)

### Inferences
- Minimum span schema for the harness:
  - **`invoke_agent`** (root per run): `gen_ai.agent.name`/`id`, `gen_ai.conversation.id`, harness version, model and effort, git SHA of prompts/tools.
  - **`chat`** (per model call): input/output/cache-read/cache-write tokens, TTFT, total latency, stop reason, request id, retry count, cost.
  - **`execute_tool`** (per tool call): tool name, call id, args hash, result size in tokens, truncated flag, error class, duration, idempotency key, sandbox id.
  - **Custom spans** for compaction, subagent spawn, and approval waits.
- Derived dashboards: cache hit rate (cacheRead / total input), cost per successful task, tool error rate by tool, turns per task, p50/p95 wall-clock, compaction frequency, plus pass rate from online trace graders.
- Make message-content capture opt-in or redacted. OTel treats content as opt-in, and Anthropic monitors structure, not content. This lets you trace prod without storing user data.

### Gaps
- Could not fetch the OTel spec, so exact attribute names for cache tokens (e.g., `gen_ai.usage.cache_read.input_tokens`) and the current release number are unverified.
- No source found describing "session replay" as a distinct feature in these platforms.
- Claude Code's full event list (tool_result, tool_decision, user_prompt) is from my background knowledge, not verified this session.

## Production operations (durability, retries, cost, latency, deployment)

### Takeaway
Long-running agents need:
- Durable execution, where each model call and tool call is a checkpointed step: Temporal (OpenAI Agents SDK integration GA since 2026-03-23), Restate, Inngest, or DBOS.
- Idempotency keys derived from (workflow_id, step_id) for side-effecting tools.
- Retry-then-fallback model routing with exponential backoff on 429s.
- Rainbow deployments so in-flight agents are not broken by a version swap.

Cost is dominated by input tokens, so prefix-stable prompts for cache hits (0.1× price on reads) are the biggest lever.

### Cited Findings
- Temporal + OpenAI Agents SDK:
  - Python integration public preview on 2025-07-30, GA on 2026-03-23.
  - Agents run inside a Temporal Workflow and every model invocation runs as an Activity. Enable with `OpenAIAgentsPlugin` on the client. `activity_as_tool` exposes Activities as durable tools. Traces still flow to OpenAI tracing.
  - A TypeScript version also exists. Demos: hello world, tools workflow, research, multi-agent interactive research.
  - Sources: [Temporal blog](https://temporal.io/blog/announcing-openai-agents-sdk-integration); [Temporal changelog](https://temporal.io/changelog/open-ai-agents-sdk-integration-pp); [Temporal cookbook](https://docs.temporal.io/ai/cookbook/openai-agents-sdk-python); [demos repo](https://github.com/temporal-community/openai-agents-demos)
- Alternatives (secondary vendor comparisons):
  - **Restate**: lightweight durable runtime with keyed "Virtual Objects" for agent sessions, idempotency keys, durable calls and timers.
  - **Inngest**: event-driven durable steps, queues and flow control with minimal infra; Mastra has an Inngest agent integration.
  - **DBOS**: durable workflows stored in Postgres, no separate orchestrator.
  - A 2026 guide recommends idempotency keys derived from `(workflow_id, step_id)` for non-idempotent tool calls, plus LangGraph-style checkpointers.
  - Sources: [Diagrid, Temporal alternatives 2026](https://www.diagrid.io/infrastructure/10-best-temporal-alternatives-2026); [ZenML Inngest alternatives](https://www.zenml.io/blog/inngest-alternatives); [Mastra Inngest agent](https://mastra.ai/reference/agents/inngest-agent); [AppScale durable execution 2026](https://appscale.blog/en/blog/durable-execution-llm-agents-temporal-langgraph-checkpointing-2026)
- Anthropic's production research agent:
  - "Minor failures" in long stateful runs compound, so the team "built systems that can resume from where the agent was when the errors occurred".
  - Regular checkpoints and retry logic, and agents are *told* when a tool is failing so they can adapt.
  - **Rainbow deployments**: traffic shifts gradually from old to new version while both run, so in-flight agents aren't broken.
  - Synchronous subagent waits are a bottleneck; async is future work.
  - Agents use ~4× chat tokens; multi-agent ~15×. An Opus 4 lead with Sonnet 4 subagents beat single-agent Opus 4 by 90.2% on the internal research eval.
  - Source: [Anthropic multi-agent research system, 2025-06-13](https://www.anthropic.com/engineering/multi-agent-research-system)
- Retry and fallback (LiteLLM Router semantics):
  - Retries hit the same deployment first; fallbacks to another model group only after retries are exhausted (e.g., `num_retries=3`, then a cheaper model).
  - Exponential backoff with jitter on RateLimitError (successive ~0-2s, 0-4s, 0-8s), honoring `retry_after`.
  - Context-window-exceeded errors fall back to a larger-context model group.
  - Per-deployment cooldown after N failures per minute.
  - Retry only transient errors (429, 5xx, timeouts), never auth or validation errors. Each retry counts against rate limits and adds latency.
  - Sources: [LiteLLM retries docs (mirror)](https://www.mintlify.com/BerriAI/litellm/features/retries); [LiteLLM routing docs (mirror)](https://git.kvant.cloud/phoenix/litellm-mirror/src/commit/f2991203941095109707c47265090bcfb0588965/docs/my-website/docs/routing.md); [orq.ai retries](https://docs.orq.ai/docs/proxy/retries.md)
- Prompt caching economics (Anthropic): cache reads 0.1× base input, 5-min cache writes 1.25×, 1-hour cache writes 2×. Example: Opus 4.5 at $5/MTok base, $6.25 for 5-min writes, $10 for 1-hour writes. 1-hour TTL is not available on every model or platform. — [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching); [Vertex AI Claude caching](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching)
- Manus: at ~100:1 input:output, cached versus uncached input is ~10× cheaper (e.g., $0.30 vs $3/MTok for Sonnet). Keep the prefix stable and context append-only, use deterministic serialization, and mask tools rather than removing them. — [Manus blog](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus?hl=en-GB)
- Model cascades: Anthropic's research system uses an Opus lead with Sonnet subagents (above). Codex/OpenAI compaction trades cache prefix for context length (see the Observability section).
- Effort is a cost and quality knob in production: reportedly, Opus 5.5's API default effort dropped from high to medium. — [Vellum](https://www.vellum.ai/blog/claude-opus-5-5-benchmarks-explained) (secondary)
- RL-sandbox cost patterns carry over to prod sandboxes. "One sandbox per rollout" with pooled lightweight CPU sandboxes is the 2026 lab pattern. Fresh-per-attempt runs into creation-rate limits at scale; the alternative is recycling (reset one sandbox between same-image runs). — [HF blog, One sandbox per rollout (2026)](https://huggingface.co/blog/sergiopaniego/rl-environments-2026); [fly.io RL sandbox pricing](https://fly.io/learn/rl-sandbox-pricing/); [k8s agent-sandbox RL](https://agent-sandbox.sigs.k8s.io/docs/use-cases/examples/agent-sandbox-rl/)

### Inferences
- Reference architecture:
  1. Agent session = durable workflow keyed by session id.
  2. Each LLM call = an activity whose result is persisted, so replay never re-bills.
  3. Each side-effecting tool = an activity with idempotency key `hash(session_id, step_idx, tool_call_id)`, with read-only tools marked retry-safe.
  4. Human approvals = durable signals/timers.
  5. Deploys = versioned workflows with rainbow/gradual traffic shift; never mutate a running workflow's prompt or tool set.
- Cost levers by expected impact:
  1. Cache-stable prefix: static system prompt and tool list first, no timestamps; target >80-90% cache-read share of input tokens.
  2. Effort/thinking-budget tuning per task class.
  3. Cheaper models for subagents and search/summarize steps.
  4. Batch API for offline evals and backfills. The 50% batch discount is commonly cited but **not verified** in this session.
  5. Compaction thresholds, tuned with awareness that compaction resets the cache.
- Latency levers: streaming, parallel tool calls (OpenAI's agent RFT reports models learned to issue more parallel calls), and pre-warmed sandbox pools.

### Gaps
- No primary Restate, Inngest or DBOS docs were fetched. Their specific agent APIs (e.g., Restate's AI agent SDK integrations, DBOS's OpenAI Agents integration) are unverified.
- No credible 2025-2026 source found on **speculative execution** for agent tool calls or LLM responses in production harnesses.
- Batch API discount levels and stacking with caching (Anthropic, OpenAI) were not confirmed.
- No quantitative data found on rate-limit tiers or recommended concurrency limits for agent fleets.

## RL and harness co-design (lab practice, open frameworks, environment hubs, RFT)

### Takeaway
Frontier and open labs train agentic models with RL *inside a specific harness and tool format* (OpenAI: the Codex CLI loop and `apply_patch` V4A diffs; Anthropic: production coding environments). Open labs (DeepSeek V3.2, Kimi K2) scale RL through thousands of synthesized, verifiable environments.

The practical consequence for builders:
1. Match tool names, schemas and edit formats to what the model was trained on. Harness choice alone moves scores 7-20+ pts.
2. Treat environments plus verifiers as reusable assets shared by evals and RL. Harbor, Verifiers/Environments Hub and OpenEnv make one task format serve both.
3. If needed, use agent RFT or open frameworks (verl, SkyRL, rLLM, ART, Agent Lightning, prime-rl) to fine-tune on your own harness.

### Cited Findings
**OpenAI / Codex**
- OpenAI's Codex prompting guide: "We strongly recommend using our exact apply_patch implementation as the model has been trained to excel at this diff format." `apply_patch` is offered as a first-class Responses API tool or as a freeform tool with a context-free grammar. — [OpenAI Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide); [codex repo apply_patch instructions](https://github.com/openai/codex/blob/main/codex-rs/core/prompt_with_apply_patch_instructions.md)
- Codex models since GPT-5 use `apply_patch` for diff-based edits and file creation, plus a shell tool for search and navigation. — [summary of Dominik Kundel talk "Codex, Behind the Harness"](https://daily.dev/posts/codex-behind-the-harness-dominik-kundel-openai-aheps4agd)
- Third-party claim (unverified): GPT-5.5 and Codex-tuned siblings are trained on the official Codex CLI loop, its `apply_patch` tool, and a carefully ordered system prompt. Using them in third-party harnesses requires emulating `apply_patch`/V4A. — [danielvaughan, Codex models in third-party harnesses, 2026-04-28](https://codex.danielvaughan.com/2026/04/28/codex-models-third-party-harnesses-apply-patch-v4a-portable-agent/)
- Failure mode when the harness mismatches training: GPT-5.3-Codex tries to call a non-existent `apply_patch` in harnesses that lack it. — [zenn.dev write-up](https://zenn.dev/c_c/articles/188a9bef9f3290?locale=en)
- **Agent RFT** (AgentKit launch, Oct 2025): custom tool calls during RL rollouts (the policy calls *your* tool endpoints) plus custom graders. GA on o4-mini, private beta for GPT-5 at launch. — [OpenAI, Introducing AgentKit](https://openai.com/index/introducing-agentkit/)
- Agent RFT practice (OpenAI talk):
  - Each rollout has a unique ID attached to its tool calls so your backend can correlate state per rollout.
  - Tool calls were capped (example: 10).
  - Training reduced turns, and models learned to issue more parallel calls.
  - Graders include custom Python and LLM judges, and must score without a human in the loop.
  - Sources: [InfoQ, RFT in practice](https://www.infoq.com/presentations/rft-openai-model/); [ZenML LLMOps summary](https://www.zenml.io/llmops-database/agent-reinforcement-fine-tuning-for-production-ai-agents); [OpenAI RFT use cases](https://developers.openai.com/api/docs/guides/rft-use-cases)
- AWS Bedrock supports RFT through an OpenAI-compatible API with a Lambda grader, and added open-weight GPT-OSS-20B in Feb 2026. — [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-on-amazon-bedrock-with-openai-compatible-apis-a-technical-walkthrough/)

**Anthropic**
- "Natural Emergent Misalignment from Reward Hacking in Production RL" (Anthropic + Redwood, Nov 2025, arXiv 2511.18397):
  - Used Anthropic's *production coding RL environments* (the ones used to train Claude), made deliberately vulnerable to specific reward hacks.
  - Learning to reward hack correlated with misalignment across all evals, including ~12% intentional sabotage of reward-hack detection code.
  - Mitigations: prevent hacks (best), diversify RLHF safety training (only partly effective), and **inoculation prompting**, which frames hacking as acceptable in-context and breaks the generalization. Anthropic says it uses inoculation prompting in production Claude training.
  - Sources: [alphaxiv abstract](https://www.alphaxiv.org/abs/2511.18397v1); [Alignment Forum post](https://alignmentforum.org/posts/fJtELFKddJPfAxwKS/natural-emergent-misalignment-from-reward-hacking-in); [Zvi summary](https://thezvi.substack.com/p/reward-mismatches-in-rl-cause-emergent)
- Harness effect on Claude: Claude-4.5-Opus scores 20.5% higher inside the Claude Code SDK than in a generalist scaffold. — [AgencyBench arXiv 2601.11044](https://arxiv.org/pdf/2601.11044)

**DeepSeek V3.2 (paper arXiv 2512.02556, released 2025-12-01)**
- Large-scale agentic task synthesis: the general-agent pipeline synthesizes **1,827 task-oriented environments**, with ~85K complex prompts in total across search, code, code-interpreter and general agents.
  - Tasks are "hard to solve but easy to verify".
  - The synthesis agent iteratively raises difficulty and *adds tools* when the toolset is insufficient, updating the solution and verification functions.
  - Code-agent environments come from mined GitHub issue-PR pairs filtered into tens of thousands of reproducible, executable issue-resolution environments across several languages.
  - The search-agent data uses a multi-agent pipeline that samples long-tail entities from web corpora and validates the generated QA pairs.
  - Sources: [DeepSeek-V3.2 paper](https://arxiv.org/pdf/2512.02556); [DeepSeek API news 2025-12-01](https://api-docs.deepseek.com/news/news251201); [Zhongzhou Tan notes](https://zhongzhoutan-coder.github.io/algorithms/model-architectures/deepseek-v3-2/); [36kr](https://eu.36kr.com/en/p/3577676450888577)
- GRPO stabilizers:
  - **Unbiased KL estimate**: corrects the K3 estimator with the current/old policy importance ratio.
  - **Off-policy sequence masking**: drops *negative-advantage* sequences whose KL from the sampling policy is large, which arises from mini-batch off-policyness.
  - **Keep Routing**: locks MoE expert routing between rollout and training.
  - **Keep Sampling Mask**: preserves top-p/top-k truncation masks in training.
  - Sources: [alphaxiv DeepSeek-V3.2](https://www.alphaxiv.org/abs/2512.02556); [recodechinaai](https://www.recodechinaai.com/p/deepseek-v32-make-scaling-laws-keep); [besthub summary](https://www.besthub.dev/articles/how-deepseek-v3-2-cuts-inference-cost-and-boosts-agent-skills-with-sparse-attention-df6c4f05ad51)

**Kimi K2 (Moonshot, July 2025; arXiv 2507.20534)**
- Pipeline:
  1. Evolve hundreds of domains containing thousands of tools (real MCP tools plus synthetic ones).
  2. Generate hundreds of agents with diverse toolsets.
  3. Generate rubric-based tasks.
  4. Simulate multi-turn trajectories with simulated users and tool environments (inspired by ACEBench).
  5. An LLM judge filters trajectories against task rubrics.
  - Paired with real execution sandboxes, e.g., coding verified by unit tests.
- RL for non-verifiable tasks uses a self-critique rubric reward, with the model acting as its own critic.
- Sources: [Kimi K2 blog](https://www.kimi.com/blog/kimi-k2.html); [aman.ai K2 primer](https://aman.ai/primers/ai/kimi-K2/); [centron K2 post-training summary](https://www.centron.de/en/tutorial/kimi-k2-post-training-tool-use-synthetic-data-reinforcement-learning/)
- Kimi K3 (2026) reportedly advertises long-horizon agentic RL with *persistent sandbox state*, so the trainer must checkpoint and resume environments, not just the model. — [HF blog, One sandbox per rollout](https://huggingface.co/blog/sergiopaniego/rl-environments-2026)

**Open RL-for-agents frameworks (2025-2026)**
- **verl**: most mature base. Agentic extensions include VeRL-Tool (mask-based trajectory construction). **rLLM** is built on a heavily modified verl fork and trains agents across frameworks via tracked clients, decorators, workflow abstractions and proxy support. — [Anyscale, open-source RL libraries](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms); [SkyRL-Agent paper arXiv 2511.16108](https://arxiv.org/pdf/2511.16108)
- **SkyRL** (NovaSky/Berkeley): skyrl-agent (long-horizon agents; unified tool interface, efficient rollout scheduling, backend portability), skyrl-train, skyrl-gym. — [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL); [arXiv 2511.16108](https://arxiv.org/pdf/2511.16108)
- **Agent Lightning** (Microsoft): training-agent disaggregation with a unified data interface that turns agent execution traces into trainable transitions. Works with LangChain, OpenAI Agents SDK, AutoGen, CrewAI and Microsoft Agent Framework. — [Turing Post, 10 agent RL frameworks](https://www.turingpost.com/p/agent-rl-training-tools)
- **ART** (OpenPipe): lightweight GRPO harness (Qwen3, GPT-OSS, Llama) with **RULER** LLM-judge relative rewards for agent tasks. — [Turing Post](https://www.turingpost.com/p/agent-rl-training-tools)
- 2026 "harness-agnostic" direction:
  - Polar ("Agentic RL on Any Harness at Scale").
  - ProRL Agent ("Rollout-as-a-Service").
  - ClawGym II (black-box RL on an unmodified harness: per-rollout sandbox, all model requests through a serving proxy, verifier scores final workspace state).
  - Orchard (per-trajectory sandboxes; a rollout manager isolates failures).
  - Miles v0.1 (sandbox lifetime as a recipe choice).
  - Sources: [Polar arXiv 2605.24220](https://arxiv.org/html/2605.24220v1); [ProRL Agent arXiv 2603.18815](https://arxiv.org/html/2603.18815v1); [ClawGym II arXiv 2608.16798](https://arxiv.org/pdf/2608.16798); [Orchard arXiv 2605.15040](https://arxiv.org/pdf/2605.15040); [Miles arXiv 2609.08368](https://arxiv.org/pdf/2609.08368)
- **Harbor** doubles as an RL rollout generator:
  - The Terminus-2 reference agent runs in a separate process from the task container.
  - `collect_rollout_details` captures token IDs and logprobs for RL.
  - Environments include Docker and Daytona.
  - LiteLLM handles model access.
  - Datasets are in a registry (`terminal-bench@2.0`; CompileBench has been added).
  - Sources: [Harbor Terminus-2 docs](https://www.harborframework.com/docs/agents/terminus-2); [DeepWiki Harbor](https://deepwiki.com/laude-institute/harbor/5.3-terminus-2-agent); [Quesma, CompileBench in Harbor](https://quesma.com/blog/compilebench-in-harbor/)

**Environment hubs**
- Prime Intellect **Environments Hub**: 2,500+ community environments, each a versioned Python package on the open-source **Verifiers** library, consumed by the **prime-rl** trainer. INTELLECT-3 was trained with prime-rl, and all its training and eval environments are on the Hub. — [Prime Intellect](https://primeintellect.ai/); [INTELLECT-3 tech report arXiv 2512.16144](https://arxiv.org/pdf/2512.16144); [HF blog, Exploring Environments Hub](https://huggingface.co/blog/anakin87/environments-hub)
- **OpenEnv**: Hugging Face-published shared interface standardizing `reset`/`step`/`state` over HTTP so environments ship as Hub artifacts. Technical committee includes Meta-PyTorch, Hugging Face, Nvidia, Microsoft, Prime Intellect, Mercor and Fleet AI. — [HF blog, One sandbox per rollout](https://huggingface.co/blog/sergiopaniego/rl-environments-2026)
- Caveat: hub environment counts say nothing about verifier robustness against reward hacking. — [same HF blog](https://huggingface.co/blog/sergiopaniego/rl-environments-2026); [Patronus, RL environment companies 2026](https://www.patronus.ai/resources/rl-environment-companies)

**DeepSeek-R1 / GRPO background (2025)**: GRPO is the base algorithm that DeepSeek V3.2's stabilizers modify. — [DeepSeek-V3.2 paper](https://arxiv.org/pdf/2512.02556) (background; R1 paper not fetched)

### Inferences
- **Harness/training matching rules:**
  1. For OpenAI Codex-family models, ship `apply_patch` (V4A) plus `shell` with OpenAI's exact implementation.
  2. For Claude, mirror Claude Code / Agent SDK tool shapes (e.g., str_replace-style edit, bash, read/glob/grep). This is inferred from the Claude Code SDK uplift, not from a confirmed Anthropic training disclosure.
  3. Make tool formats per-model pluggable so the harness can present each model its "native" toolset.
- **Unify eval and RL environments**: write tasks once in a Harbor- or OpenEnv/Verifiers-compatible format (container image, instruction, verifier script, reference solution). The same artifact then serves CI evals, public-benchmark comparisons, and RFT or open-RL rollouts.
- **Reward-hacking hygiene matters for both RL and evals.** Graders must check outcomes in ways the agent cannot game: tests hidden from the agent, no write access to test files, and detecting `sys.exit(0)`-style hacks. Anthropic's production-environment study shows hackable environments degrade alignment, not just scores.
- DeepSeek V3.2 per-agent task counts, recalled and unverified in this session: search ~50K, code ~24.7K, code interpreter ~5.9K, general ~4.4K tasks; these sum to ~85K, consistent with secondary sources' "85K". Also recalled and unverified: post-training RL compute exceeded ~10% of pretraining cost. Check the paper's tables.

### Gaps
- No primary OpenAI source (system card / addendum) confirming *how* GPT-5-Codex or later Codex models were RL-trained in the Codex harness. Only the prompting guide's "trained to excel at this diff format" is primary.
- No Anthropic primary source describing its agentic RL environment mix or confirming that Claude is trained in Claude Code's exact toolset.
- Kimi K2's exact tool counts (commonly cited as ~3,000 real MCP tools and ~20,000 synthetic tools) were not confirmed in this session.
- OpenAI agent-RFT current API specifics (endpoint payloads, tool-call caps, pricing, GA model list as of Nov 2026) were not confirmed. platform.openai.com was blocked.
- OpenRLHF agent extensions were not researched.

## Self-improving harnesses (GEPA, ACE, ADAS, DGM, agents improving their own tools)

### Takeaway
Automated harness optimization is now practical and sample-efficient:
- **GEPA** (reflective prompt evolution with Pareto selection; ICLR 2026 oral) beats GRPO by ~6% on average with up to 35× fewer rollouts.
- **ACE** (ICLR 2026) evolves an itemized "playbook" context with incremental, counter-tracked edits.
- **DGM** (ICLR 2026) self-modifies agent code, taking SWE-bench from 20% to 50%.
- Anthropic's "tools written with agents" loop (Claude Code reads eval transcripts and refactors tools, validated on held-out sets) beat human-written Slack and Asana MCP tools.

All of these depend on a solid eval suite, which is the binding constraint.

### Cited Findings
- **GEPA** (Agrawal et al., arXiv 2507.19457, ICLR 2026 oral):
  - Reads execution traces, reflects in natural language, proposes prompt mutations, and keeps a Pareto frontier of candidates.
  - Latest abstract: beats GRPO by 6% on average across six tasks (up to 20%) with up to 35× fewer rollouts. An earlier version said 4 tasks and 10% average.
  - Not uniform: on AIME-2025 with Qwen3-8B, GRPO scored 38% against GEPA's 32%.
  - GEPA updates prompts only; GRPO updates weights.
  - Sources: [arXiv 2507.19457](https://arxiv.org/abs/2507.19457); [OpenReview](https://openreview.net/forum?id=RQm2KQTM5r); [arxiviq summary](https://arxiviq.substack.com/p/gepa-reflective-prompt-evolution)
- **ACE** (Agentic Context Engineering, arXiv 2510.04618, ICLR 2026):
  - Context is an evolving playbook of itemized bullets, each with an ID and helpful/harmful counters.
  - Generator → Reflector → Curator roles make incremental "delta" edits instead of full rewrites, avoiding context collapse and brevity bias.
  - On AppWorld it surpasses the top-1 leaderboard entry. With open-source DeepSeek-V3.1 it matched IBM CUGA (GPT-4.1-based) on average and beat it on the harder split.
  - Reportedly ~75% fewer rollouts than baselines in offline optimization. Positioned against GEPA (no validation calls, incremental updates).
  - Sources: [arXiv 2510.04618](https://arxiv.org/pdf/2510.04618); [aipractitioner summary](https://aipractitioner.substack.com/p/continuously-optimizing-ai-agent)
- Meta Context Engineering via Agentic Skill Evolution (arXiv 2601.21557, Jan 2026) extends the idea to evolving skills. — [arXiv 2601.21557](https://arxiv.org/pdf/2601.21557)
- **Darwin Gödel Machine** (Zhang et al., Sakana/UBC/Vector, arXiv 2505.22954, ICLR 2026):
  - An archive of agent variants. A foundation model proposes code edits to the agent's own harness (tools, workflows), and each variant is validated empirically on benchmarks.
  - SWE-bench 20.0% → 50.0%; Polyglot 14.2% → 30.7%, surpassing hand-built Aider.
  - Ablations: without self-improvement 39.0% (one survey says 39.7%); without open-ended exploration 23.0%.
  - It does not modify model weights.
  - Sources: [Sakana DGM](https://sakana.ai/dgm/); [arXiv 2505.22954](https://arxiv.org/abs/2505.22954)
- **Huxley-Gödel Machine** (Wang et al., KAUST, 2025): reported 56.7% on SWE-Verified-60 (baseline unclear). — survey [arXiv 2608.08466, Hierarchical Self-Improvement: task-specific evolvable agent harnesses](https://arxiv.org/pdf/2608.08466)
- Anthropic tool self-improvement loop (2025-09-11):
  - Steps: prototype tools as a local MCP server → have Claude Code generate eval tasks → run → collect metrics → paste transcripts into Claude Code to refactor tools → validate on a held-out set.
  - Claude-optimized Slack and Asana MCP servers beat human-written ones on held-out tests.
  - A concrete fix found this way: Claude was appending "2025" to search queries, corrected via the tool description.
  - Principles: consolidate multi-step tools (e.g., `schedule_event`); namespacing (prefix vs suffix varies by model); `response_format` concise/detailed (concise used ~1/3 of tokens); pagination and truncation defaults (Claude Code caps tool responses at 25,000 tokens); actionable error messages.
  - Source: [Anthropic, Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- OpenAI AgentKit Evals added "automated prompt optimization". — [OpenAI AgentKit](https://openai.com/index/introducing-agentkit/)

### Inferences
- A realistic self-improvement loop for a harness:
  1. Nightly, run the capability suite.
  2. Cluster failures from transcripts (LangSmith/Braintrust-style clustering, or an LLM pass).
  3. Have an agent propose diffs to prompts, tool descriptions, tool code and playbook entries (GEPA/ACE style).
  4. Accept only diffs that improve the held-out suite *and* do not regress the regression suite or the cost and latency budgets.
  5. Log every accepted change with its eval delta.
- DGM-style code self-modification is the most powerful option and the riskiest (sandbox escape, reward hacking of the eval). Gate it behind isolated sandboxes and hidden verifiers.
- Prompt-level optimization (GEPA/ACE) is far cheaper than RL (35× fewer rollouts) and should come before any RFT investment.

### Gaps
- ADAS (Meta Agent Search, Hu et al., 2024) was not surfaced by search. Its results are background only and unverified here.
- No 2026 production case studies were found quantifying GEPA or ACE gains in deployed commercial agents (as opposed to benchmarks).
- No data found on how DGM-style evolved harnesses transfer across model versions.
