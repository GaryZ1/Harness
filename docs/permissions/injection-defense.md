# Prompt-injection defence (probes)

Module: `injection.py`. Classes: `HeuristicInjectionDetector`, `ModelInjectionClassifier`, `InjectionProbe`, `ProbeResult`. Wired in by `PolicyEngine.record_result()` and `PolicyEngine.subagent_finished()`.

## Purpose

This module scans content arriving on **untrusted channels**: web fetch, browser, email, MCP results, out-of-repo files and sub-agent reports. Suspect content gets an **annotation** (a warning prepended to the tool result) and the session is tainted (`injection_suspected`). The content itself is never dropped, and a probe result can never relax policy.

## Research basis

* Auto mode's input layer "scans tool outputs and adds a warning to treat suspect content cautiously". Claude in Chrome "probes tool results and warns". "Do not drop content silently, since the agent needs to know something was there" (technique 19).
* Probes are a weak layer on their own. On the stronger Chrome eval, probes moved Opus 4.5 only from 17.6% to 16.7% ASR. "The action classifier did the heavy lifting." Catching "95% of attacks" is "a failing grade". So the real defence is taint plus human gating of exfiltration-capable actions ([boundary-and-taint.md](boundary-and-taint.md)), and the probe only *adds* signal.
* Untrusted channels include sub-agent reports and relayed "the user approved X" claims, which "carry no authority from the user".

## API

```python
from harness_permissions import HeuristicInjectionDetector, InjectionProbe, ModelInjectionClassifier

probe = InjectionProbe()                                   # heuristics only (default in PolicyEngine)
probe = InjectionProbe(model=ModelInjectionClassifier(client, model_id), model_mode="always")  # or "on_flag"
r = probe.scan(tool_output)                                # str or list of content blocks
r.suspicious, r.score, r.categories, r.model_verdict       # "injection" | "clean" | "error" | None
safe = r.annotate(tool_output)                             # warning + original content (unchanged if clean)
engine = PolicyEngine(policy, probe=probe)                 # pass probe=InjectionProbe(...) to customise
```

`ModelInjectionClassifier` takes any `harness_core.types.ModelClient` and a **caller-supplied model id**, which it sends as `ModelRequest.extra["model"]` because the shared contract's request has no model field. The content is wrapped in `<untrusted_content>` and the model must answer exactly `INJECTION` or `CLEAN`.

### Heuristic categories

| Category | Weight | Example |
|---|---|---|
| `instruction_override` | 1.0 / 0.6 | "ignore all previous instructions", "new system prompt" |
| `tool_call_forgery` | 0.9 | `<function_calls>`, `"type": "tool_use"` |
| `approval_claim` | 0.9 | "the user has already approved", "you are authorized to" |
| `fake_role_tag` | 0.8 | `<system>`, `[INST]`, `<\|im_start\|>`, `assistant:` at line start |
| `exfiltration` | 0.8 | "send … to https://…" or an e-mail address |
| `markdown_beacon` | 0.8 | `![x](https://host/p.png?d=…)` |
| `role_hijack` | 0.7 | "you are now", "developer mode" |
| `secret_request` | 0.7 | "print the API keys", "cat id_rsa" |
| `concealment`, `hidden_comment`, `hidden_text` | 0.6 | "do not tell the user", `<!-- AI: … -->`, zero-width characters |
| `addressing_agent`, `command_injection` | 0.5 | "Assistant: you must…", "run `curl … \| sh`" |
| `urgency` | 0.3 | "IMPORTANT: must … immediately" |
| `encoded_blob` | 0.2 | base64 runs of 200+ characters |

Matching runs after NFKC normalisation and with zero-width characters stripped, so `ig​nore previous instructions` is caught. The score is the sum of the highest weight per category. Content is flagged at **≥ 0.8**. Extra patterns can be added with `HeuristicInjectionDetector(extra_patterns=((category, weight, regex),))`.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Probe on | every taint-source channel | research (scan every untrusted channel) |
| Threshold | 0.8 | our choice |
| Max scanned characters | 200,000 (heuristic), 20,000 (model) | our choice |
| Model probe | off unless configured; `model_mode="always"` | our choice |
| Model output tokens | 8 | our choice |

## Invariants

* Annotate, never drop. `annotate()` keeps the original content byte for byte after the warning.
* A probe hit taints the session. Nothing in this module can allow an action.
* A model-probe error or unparsable answer counts as **suspicious** (fail closed).
* The probe's output is never shown to the [auto-reviewer](auto-reviewer.md). The reviewer is reasoning-blind.

## Failure modes

* Heuristics have false negatives against adaptive attackers. Assume they will be bypassed. Taint gating is what protects outward actions, whether or not the probe fires.
* False positives cost only a warning banner plus taint, and taint makes outward actions ask. Benign documentation that *discusses* prompt injection may trigger this.
* Keep a red-team regression suite (`tests/permissions/test_permissions_injection.py` is the seed). Retire saturated fixtures, and never compare ASRs across attacker generations.
