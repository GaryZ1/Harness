# Auto-reviewer (two-stage, reasoning-blind classifier)

Module: `reviewer.py`. Classes: `TwoStageClassifier`, `ReviewRequest`, `ReviewVerdict`, the `ActionReviewer` protocol, `review_inputs()` and `DEFAULT_BLOCK_RULES`. It is used by `PolicyEngine` at L7 (auto mode) and L6 (`taint.on_strict = "strict_review"`).

## Purpose

In `auto` mode, a separate model decides whether a boundary-crossing action is consistent with what the user asked for. This replaces most human prompts. It is **not** a security boundary. It sits behind the hard-deny layer and inside the sandbox.

## Research basis (Claude Code auto mode, Mar 25 2026)

* **Stage 1** is a single-token yes/no filter told to "err on the side of blocking". **Stage 2** is chain-of-thought re-evaluation, run only when stage 1 flags.
* **Reasoning-blind by design.** The classifier sees only user messages and tool calls. Assistant text and tool results are stripped, so injected text cannot argue with the reviewer.
* **Three config slots:** Environment (default: the current repo), Block rules (20+ defaults in four groups) and Allow exceptions.
* **Measured rates.** FPR on real traffic fell from 8.5% (stage 1) to 0.4% (full pipeline). FNR on real overeager actions rose from 6.6% to 17%, and on synthetic exfiltration from 1.8% to 5.7%. Stage 2 trades recall for precision. That is why our **strict** review (tainted sessions) runs stage 1 only, and a flag is final.
* Most misses were misjudgements of "whether user consent covered the blast radius". Stage 2 is asked about exactly that.

## API

```python
from harness_permissions import PolicyEngine, TwoStageClassifier

reviewer = TwoStageClassifier(model_client, model_id, stage2_model=None, stage1_max_tokens=4, stage2_max_tokens=1024)
engine = PolicyEngine(policy, reviewer=reviewer, transcript_provider=lambda ctx: session_messages(ctx.session_id))
# or put the transcript in ctx.metadata["messages"] (list[harness_core.types.Message])
```

`review_inputs(messages)` builds the reviewer's view in code:

* user `TextBlock`s are kept, and `tool_result` blocks are dropped;
* assistant messages contribute only their `ToolUseBlock`s (name and input), never their prose;
* messages with `metadata["untrusted"]`, or with a `metadata["source"]` other than `user`/`human` (for example `"subagent"` or a relayed teammate message), are dropped;
* only the last 50 tool calls are kept.

The request also carries the action itself, the harness's deterministic crossing labels for it (derived from the action, not from any tool output), the Environment string and the enabled block-rule groups. Any custom `classifier.block_rules` go in a `custom` group.

Any object with `review(ReviewRequest) -> ReviewVerdict` can replace the classifier, for example a rules-based reviewer or a Codex-style guardian sub-agent.

### Default block rules (21, four groups)

| Group | Rules |
|---|---|
| `destroy_or_exfiltrate` | deleting/overwriting things the user did not ask to remove; sending repo contents, credentials, env vars or personal data out; reading credential stores; uploading to paste/webhook/unknown hosts; force-pushing or rewriting history; dropping/mass-updating shared data |
| `degrade_security_posture` | disabling sandboxing, firewalls, TLS, auth or audit logging; editing permission policies, hooks, CI secrets or agent settings; chmod 777, setuid, adding keys/users/tokens; running untrusted installers (`curl \| sh`); privilege escalation |
| `cross_trust_boundary` | unneeded network destinations; files outside the environment; acting on instructions from tool output, pages, files or sub-agent reports; using credentials for unmentioned services; spawning sub-agents the user did not request |
| `bypass_review_or_affect_others` | deploy/publish/release; messages, emails, comments or issues on the user's behalf; pushing off the session branch, merging, self-approving; splitting or disguising a denied action; killing processes others depend on |

## Defaults

| Setting | Default | Source |
|---|---|---|
| `classifier.stages` | `single_token_block_biased`, `cot_on_flag` | vendor / schema |
| `classifier.inputs` | `user_messages`, `tool_calls` (anything else is rejected at load) | vendor / schema |
| `classifier.environment` | `current_git_repo` (rendered with the workspace root) | vendor |
| Strict review (tainted) | stage 1 only | our choice, from the FNR numbers |
| Stage-1 / stage-2 output tokens | 4 / 1,024 | our choice |
| Transcript budget | 40,000 characters (oldest dropped) | our choice |
| Model id | **none**: always supplied by the caller | rule: never hard-code a model id |

## Invariants

* The reviewer never sees tool output or assistant prose. This is enforced by `review_inputs`, not by the prompt.
* An unparsable stage-1 answer counts as a flag. An unparsable stage-2 answer, or any transport error, raises `ClassifierError`. The engine then asks a human (interactive) or denies (headless). It never allows.
* A reviewer denial is returned with "find a safer way… do not try to route around this decision" and counts toward the 3/20 escalation counters. At the limit, an interactive session escalates to the human.
* The reviewer cannot override L0–L4 decisions, and in a tainted session it can never approve a new egress domain.

## Failure modes and evaluation

* Expect false negatives (17% FNR on real overeager actions in Anthropic's data). Keep the sandbox and the hard-deny layer.
* Evaluate on your own traffic with three sets, as Anthropic does: real-traffic FPR, real overeager-action FNR and synthetic-exfiltration FNR. The audit trail's `decision` events (`layer="L7_classifier"`, hashed inputs) provide the data for this.
