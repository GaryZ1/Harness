"""Graders: end-state (deterministic, first), transcript, integrity, LLM-judge rubric, composite.

Importing this package registers every built-in kind:

==================  ===========================================  ==============
kind (aliases)      class                                        deterministic
==================  ===========================================  ==============
file_exists         :class:`~.state.FileExistsGrader`            yes
file_regex          :class:`~.state.FileRegexGrader`             yes
file_equals         :class:`~.state.FileEqualsGrader`            yes
json_assert         :class:`~.state.JsonAssertGrader`            yes
answer              :class:`~.state.AnswerGrader`                yes
command/state_check :class:`~.state.CommandGrader`               yes
unit_tests          :class:`~.state.UnitTestsGrader`             yes
diff_scope          :class:`~.state.DiffScopeGrader`             yes
transcript/tool_calls :class:`~.transcript.TranscriptGrader`     yes
integrity/no_tamper :class:`~.integrity.IntegrityGrader`         yes
llm_rubric/llm_judge :class:`~.judge.LLMJudgeGrader`             no
composite           :class:`~.composite.CompositeGrader`         if children are
==================  ===========================================  ==============
"""

from .base import GradeContext, Grader, GraderDeps, GradeResult, build_grader, build_graders, grader_kinds, register_grader
from .composite import CompositeGrader, TaskScore, safe_grade, score_task
from .integrity import HackSignal, IntegrityGrader, detect_hack_signals, hack_pattern_counts, has_tamper
from .judge import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, Criterion, LLMJudgeGrader, neutralise, parse_verdicts
from .state import (
    AnswerGrader,
    CommandGrader,
    DiffScopeGrader,
    FileEqualsGrader,
    FileExistsGrader,
    FileRegexGrader,
    JsonAssertGrader,
    UnitTestsGrader,
    glob_match,
    json_pointer,
)
from .transcript import TranscriptGrader

__all__ = [
    "AnswerGrader",
    "CommandGrader",
    "CompositeGrader",
    "Criterion",
    "DiffScopeGrader",
    "FileEqualsGrader",
    "FileExistsGrader",
    "FileRegexGrader",
    "GradeContext",
    "GradeResult",
    "Grader",
    "GraderDeps",
    "HackSignal",
    "IntegrityGrader",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "JsonAssertGrader",
    "LLMJudgeGrader",
    "TaskScore",
    "TranscriptGrader",
    "UnitTestsGrader",
    "build_grader",
    "build_graders",
    "detect_hack_signals",
    "glob_match",
    "grader_kinds",
    "hack_pattern_counts",
    "has_tamper",
    "json_pointer",
    "neutralise",
    "parse_verdicts",
    "register_grader",
    "safe_grade",
    "score_task",
]
