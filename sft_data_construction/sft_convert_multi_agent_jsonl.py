# -*- coding: utf-8 -*-
# @Date    : 2026-03-21
# @File    : sft_convert_multi_agent_jsonl.py
# @Desc    : Convert MAS error data into joint label+agent JSON SFT data.
#
# This version:
# 1) keeps no-agent failure hypotheses
# 2) groups positive supervision by error_type, because one error can involve multiple agents
# 3) for entail samples, the response contains multiple JSON objects, one per responsible agent
# 4) for neutral / contradict samples, the response contains exactly one JSON object with agents=[]
# 5) SFT response format example:
#       {"label":"A","agents":["Planner"]}
#       {"label":"A","agents":["Solver"]}

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# =========================================================
# 1. Error taxonomy (14 error types only)
# =========================================================

FM_DESC = {
    # FM-1.x - Task Execution Errors
    "FM-1.1": "Task specification deviation - Agent deviates from specified task requirements.",
    "FM-1.2": "Role specification deviation - Agent acts outside its designated role.",
    "FM-1.3": "Add redundant steps - Agent adds unnecessary or duplicate steps.",
    "FM-1.4": "Remove conversation history - Agent ignores or removes important context from previous turns.",
    "FM-1.5": "Remove termination conditions - Agent fails to define proper stopping criteria, leading to loops or unfinished tasks.",

    # FM-2.x - Communication & Coordination Errors
    "FM-2.1": "Repeat handled tasks - Agent redundantly handles already completed tasks.",
    "FM-2.2": "Make request ambiguous - Agent provides unclear or confusing instructions to other agents.",
    "FM-2.3": "Deviate from main goal - Agent pursues objectives unrelated to the main task.",
    "FM-2.4": "Hide important information - Agent withholds crucial information needed by other agents.",
    "FM-2.5": "Ignore other agents - Agent fails to consider input, corrections, or questions from other agents.",
    "FM-2.6": "Inconsistent reasoning - Agent's logic contradicts its own previous statements.",

    # FM-3.x - Quality & Verification Errors
    "FM-3.1": "Premature termination - Agent stops or declares the task complete before all requirements are met.",
    "FM-3.2": "Remove verification steps - Agent skips necessary validation or testing steps.",
    "FM-3.3": "Incorrect verification - Agent performs flawed or wrong verification.",
}

ALL_ERROR_CODES = list(FM_DESC.keys())

# no-agent hypotheses
FM_TEMPLATE_NO_AGENT: Dict[str, str] = {
    "FM-1.1": "An agent deviated from the specified task requirements in the trajectory.",
    "FM-1.2": "An agent acted outside its designated role in the trajectory.",
    "FM-1.3": "An agent introduced redundant or unnecessary steps in the trajectory.",
    "FM-1.4": "An agent overlooked important context from earlier conversation history in the trajectory.",
    "FM-1.5": "An agent failed to maintain a proper termination condition in the trajectory.",
    "FM-2.1": "An agent repeated a task that had already been handled in the trajectory.",
    "FM-2.2": "An agent made a request unclear or ambiguous for other agents in the trajectory.",
    "FM-2.3": "An agent drifted away from the main goal of the task in the trajectory.",
    "FM-2.4": "An agent omitted important information needed by other agents in the trajectory.",
    "FM-2.5": "An agent overlooked input or corrections from other agents in the trajectory.",
    "FM-2.6": "An agent used reasoning that was inconsistent with earlier statements in the trajectory.",
    "FM-3.1": "An agent ended the task before all requirements were met in the trajectory.",
    "FM-3.2": "An agent skipped necessary verification or validation steps in the trajectory.",
    "FM-3.3": "An agent performed flawed or unreliable verification in the trajectory.",
}
# =========================================================
# 2. Joint verifier prompt (JSON output)
# =========================================================

# STRICT_VERIFIER_PROMPT = """You are a careful verifier for multi-agent trajectory failure attribution.
#
# You will be given:
# 1. A trajectory
# 2. A failure hypothesis
# 3. The list of candidate agents appearing in the trajectory
#
# Your task is to jointly predict:
# - label: one of A, B, C
# - agents: the responsible agent list for this JSON object
#
# Label meanings:
# A = entail
# B = neutral
# C = contradict
#
# Decision rules:
# - Choose A when the trajectory provides clear or reasonably strong evidence that the hypothesized failure occurred, and that it negatively affected, or likely affected, the final outcome, final answer, final decision, or successful task completion.
# - Choose B when the evidence is mixed, incomplete, weak, or the impact on the final outcome is uncertain.
# - Choose C when the trajectory provides clear evidence that the hypothesis is false or inconsistent with what actually happened.
#
# Agent attribution rules:
# - A failure may be caused by one agent or multiple agents.
# - If label = A, output one JSON object for each likely responsible agent.
# - Each JSON object must contain exactly one responsible agent in the agents list, for example {"label":"A","agents":["Planner"]}.
# - Include an agent only when there is clear or reasonably strong evidence that this agent contributed to the hypothesized failure.
# - Do not include agents merely because they are mentioned near the error.
# - Do not include agents that only followed instructions from another faulty agent unless their own action also contributed to the failure.
# - Only output agent names that appear exactly in the provided candidate agent list.
# - If label = A but no responsible agent can be confidently identified, output one JSON object with agents set to an empty list.
# - If label is B or C, output exactly one JSON object with agents set to an empty list.
#
# Output format:
# Return JSON objects only.
# If there are multiple responsible agents, return multiple JSON objects, one per line.
# Each JSON object must have exactly these keys:
# {"label":"A","agents":["agent_name"]}
#
# Valid examples:
# {"label":"A","agents":["Planner"]}
# {"label":"A","agents":["Solver"]}
#
# {"label":"B","agents":[]}
#
# {"label":"C","agents":[]}
#
# Do not output explanations.
# Do not output markdown.
# Do not output extra text.
# """

STRICT_VERIFIER_PROMPT = """You are a strict JSON-only verifier for multi-agent failure attribution.

You must follow all instructions exactly.

Task:
Given a trajectory, one failure hypothesis, and a list of candidate agents, decide:
1. whether the hypothesis is supported by the trajectory;
2. which candidate agent(s), if any, are responsible.

Labels:
A = entail. The trajectory provides clear or reasonably strong evidence that the hypothesized failure occurred and likely affected the task outcome.
B = neutral. The evidence is weak, incomplete, mixed, ambiguous, or the effect on the outcome is unclear.
C = contradict. The trajectory clearly shows that the hypothesis is false or inconsistent with what happened.

Decision policy:
- Prefer B over A when the evidence is not strong enough.
- Prefer B over A when the failure was minor, corrected later, or did not clearly affect the final outcome.
- Use C only when the trajectory clearly contradicts the hypothesis.
- Do not choose A only because the hypothesis sounds plausible.
- Do not choose A only because an agent is mentioned near relevant content.

Agent attribution policy:
- Only assign agents when label is A.
- If label is A, output one JSON object for each responsible agent.
- Each A object must contain exactly one agent in the "agents" list.
- The agent name must be copied exactly from the candidate agent list.
- Do not invent agent names.
- Do not output agents outside the candidate list.
- Do not assign an agent merely because the agent appears in the trajectory.
- Do not assign an agent that only followed another agent's faulty instruction unless its own action also contributed to the failure.
- If label is A but no responsible agent can be confidently identified, output exactly one JSON object with "agents": [].
- If label is B or C, output exactly one JSON object with "agents": [].

Output format requirements:
- Output JSON only.
- Do not output markdown.
- Do not output explanations.
- Do not output analysis.
- Do not output <think>.
- Do not output any text before or after the JSON.
- Each JSON object must have exactly two keys: "label" and "agents".
- The value of "label" must be exactly one of: "A", "B", "C".
- The value of "agents" must be a list.
- For B and C, "agents" must be [].

Valid outputs:
{"label":"A","agents":["Planner"]}
{"label":"A","agents":["Solver"]}
{"label":"B","agents":[]}
{"label":"C","agents":[]}

If multiple agents are responsible, output one JSON object per line:
{"label":"A","agents":["Planner"]}
{"label":"A","agents":["Solver"]}

Now produce the final answer only."""

# Additional guidance:
# - Do not require perfect or absolute proof for A.
# - A can be chosen when the evidence overall supports the hypothesis, even if every detail is not fully explicit.
# - If the failure appears to have been minor, corrected later, or not connected to the final outcome, prefer B over A.
# - Use C only when the hypothesis is clearly contradicted by the trajectory, not merely because support is weak.

# =========================================================
# 3. Parsing helpers
# =========================================================

def extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def parse_conversation_history(history_text: str) -> List[Dict[str, Any]]:
    """
    Parse blocks like:

    Step 1 - RoleAssigner () [initialization]:
    ...
    Step 2 - Solver () [reasoning]:
    ...
    """
    pattern = re.compile(
        r"Step\s+(\d+)\s*-\s*([^(]+?)\s*\([^)]*\)\s*\[([^\]]+)\]:\s*\n(.*?)(?=\nStep\s+\d+\s*-|\Z)",
        re.DOTALL,
    )
    steps: List[Dict[str, Any]] = []
    for m in pattern.finditer(history_text):
        steps.append(
            {
                "step": int(m.group(1)),
                "agent_name": m.group(2).strip(),
                "stage": m.group(3).strip(),
                "content": m.group(4).strip(),
            }
        )
    return steps


def load_ground_truth(gt: Any) -> Dict[str, Any]:
    if isinstance(gt, dict):
        return gt
    if isinstance(gt, str):
        return json.loads(gt)
    raise TypeError(f"Unsupported ground_truth type: {type(gt)}")


def get_prompt_text(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt", [])
    if isinstance(prompt, list) and prompt:
        return prompt[0].get("content", "")
    if isinstance(prompt, str):
        return prompt
    return ""


def build_premise(query: str, history_text: str, include_query: bool = True) -> str:
    if include_query and query.strip():
        return f"QUERY:\n{query.strip()}\n\nCONVERSATION HISTORY:\n{history_text.strip()}"
    return history_text.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_agents_from_steps(steps: List[Dict[str, Any]]) -> List[str]:
    agents: List[str] = []
    seen: Set[str] = set()
    for s in steps:
        agent = s.get("agent_name")
        if agent and agent not in seen:
            seen.add(agent)
            agents.append(agent)
    return agents


# =========================================================
# 4. Hypothesis generation (no agent role)
# =========================================================

def build_hypothesis(error_type: str) -> str:
    if error_type in FM_TEMPLATE_NO_AGENT:
        return FM_TEMPLATE_NO_AGENT[error_type]
    desc = FM_DESC.get(error_type, "an unspecified error")
    return f"The trajectory contains evidence of {desc}"


def extract_positive_error_types(sample: Dict[str, Any]) -> List[str]:
    """
    Extract unique error_type values from faulty_agents.
    """
    gt = load_ground_truth(sample["reward_model"]["ground_truth"])
    gold_faults = gt.get("faulty_agents", [])

    seen = set()
    pos_errors = []
    for item in gold_faults:
        error_type = item.get("error_type")
        if error_type in FM_DESC and error_type not in seen:
            seen.add(error_type)
            pos_errors.append(error_type)
    return pos_errors


def extract_positive_fault_pairs(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract unique (error_type, agent_name) pairs from faulty_agents.
    """
    gt = load_ground_truth(sample["reward_model"]["ground_truth"])
    gold_faults = gt.get("faulty_agents", [])

    seen: Set[Tuple[str, str]] = set()
    pos_pairs: List[Dict[str, str]] = []

    for item in gold_faults:
        error_type = item.get("error_type")
        agent_name = item.get("agent_name")
        if error_type in FM_DESC and agent_name:
            key = (error_type, agent_name)
            if key not in seen:
                seen.add(key)
                pos_pairs.append(
                    {
                        "error_type": error_type,
                        "agent_name": agent_name,
                    }
                )
    return pos_pairs


def extract_positive_faults_by_error(sample: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Group gold faulty agents by error_type.

    Example:
    {
        "FM-1.1": ["Planner", "Solver"],
        "FM-3.2": ["Evaluator"]
    }

    This is the key change for multi-agent attribution: one failure hypothesis
    can have multiple responsible agents.
    """
    gt = load_ground_truth(sample["reward_model"]["ground_truth"])
    gold_faults = gt.get("faulty_agents", [])

    grouped: Dict[str, List[str]] = {}
    seen: Set[Tuple[str, str]] = set()
    for item in gold_faults:
        error_type = item.get("error_type")
        agent_name = item.get("agent_name")
        if error_type in FM_DESC and agent_name:
            key = (error_type, agent_name)
            if key not in seen:
                seen.add(key)
                grouped.setdefault(error_type, []).append(agent_name)
    return grouped


# =========================================================
# 5. Positive evidence heuristics
# =========================================================

POSITIVE_KEYWORDS = {
    "FM-1.1": ["wrong format", "javascript", "python", "requirement", "specification", "requested"],
    "FM-1.2": ["role", "critic", "evaluate", "outside"],
    "FM-1.3": ["again", "duplicate", "redundant", "repeat", "already"],
    "FM-1.4": ["ignore", "previous", "history", "context", "correction", "requirement"],
    "FM-1.5": ["loop", "unfinished", "termination", "stopping", "continue"],

    "FM-2.1": ["repeat", "already completed", "already finalized", "again"],
    "FM-2.2": ["unclear", "ambiguous", "confusing", "instruction"],
    "FM-2.3": ["unrelated", "main goal", "off-topic", "deviate"],
    "FM-2.4": ["withhold", "hide", "lacks knowledge", "limited experience", "important information"],
    "FM-2.5": ["ignore", "correction", "question", "other agent"],
    "FM-2.6": ["contradict", "inconsistent", "previous statement", "logic"],

    "FM-3.1": ["complete", "finished", "done", "task complete", "score: 100", "flawless"],
    "FM-3.2": ["skip verification", "no tests", "no validation", "without testing", "remove verification"],
    "FM-3.3": ["incorrect verification", "wrong verification", "score: 100", "flawless", "evaluation"],
}


def _keyword_score(text: str, error_type: str) -> int:
    text_low = normalize_text(text)
    score = 0
    for kw in POSITIVE_KEYWORDS.get(error_type, []):
        if kw in text_low:
            score += 1
    return score


def select_evidence_steps(
    steps: List[Dict[str, Any]],
    error_type: str,
    max_steps: int = 4,
) -> List[int]:
    """
    Select evidence steps for positive / ordinary negative samples.
    """
    if not steps:
        return []

    scored = []
    for s in steps:
        score = _keyword_score(s["content"], error_type)
        scored.append((score, s["step"]))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [step_id for score, step_id in scored if score > 0][:max_steps]

    if chosen:
        return sorted(chosen)

    return sorted([s["step"] for s in steps[:max_steps]])


# =========================================================
# 6. Contradiction heuristics
# =========================================================

CONTRADICT_KEYWORDS = {
    "FM-1.1": [
        "as requested", "according to the requirement", "followed the requirement",
        "kept the required format", "in python", "met the specification"
    ],
    "FM-1.2": [
        "according to assigned role", "within role", "as assigned", "performed its role",
        "stayed in its role"
    ],
    "FM-1.3": [
        "without duplication", "without redundant", "concise", "single final answer",
        "did not repeat"
    ],
    "FM-1.4": [
        "as mentioned earlier", "based on previous", "taking previous context into account",
        "after the correction", "considering earlier feedback", "from previous turns"
    ],
    "FM-1.5": [
        "base case", "termination condition", "stop when", "stopping criterion",
        "return when", "until condition is met"
    ],

    "FM-2.1": [
        "already completed so skip", "no need to repeat", "continue from previous result",
        "reuse the previous result", "avoid repeating"
    ],
    "FM-2.2": [
        "clear instruction", "explicit instruction", "specifically asked", "step-by-step instruction",
        "clearly stated", "explicitly stated"
    ],
    "FM-2.3": [
        "focus on the main task", "directly answer the query", "stay on task",
        "relevant to the task", "aligned with the goal"
    ],
    "FM-2.4": [
        "shared the important information", "provided all relevant information", "disclosed",
        "made it clear", "explicitly mentioned", "fully informed"
    ],
    "FM-2.5": [
        "incorporated feedback", "addressed the concern", "responded to the correction",
        "followed the suggestion", "took the feedback into account"
    ],
    "FM-2.6": [
        "consistent with earlier", "same conclusion as before", "as stated earlier",
        "remained consistent"
    ],

    "FM-3.1": [
        "all requirements are satisfied", "after verification", "completed all requested parts",
        "finished after checking", "fully completed"
    ],
    "FM-3.2": [
        "verified", "validated", "tested", "checked", "unit test", "test case", "cross-check"
    ],
    "FM-3.3": [
        "verified against expected output", "validated with test cases", "cross-checked results",
        "checked correctness", "confirmed the result"
    ],
}

CONTRADICT_STAGE_BONUS = {
    "FM-3.2": {"evaluation", "verification", "testing"},
    "FM-3.3": {"evaluation", "verification", "testing"},
    "FM-1.5": {"reasoning", "coding", "implementation"},
    "FM-2.5": {"discussion", "coordination", "evaluation", "reasoning"},
}

GENERIC_CONTRADICT_HINTS = {
    "FM-3.2": ["test", "tests", "validate", "validation", "verified", "checking", "checked"],
    "FM-3.3": ["expected output", "ground truth", "correctness", "cross-check", "comparison"],
    "FM-1.4": ["earlier", "previous", "before", "feedback", "correction"],
    "FM-2.5": ["feedback", "suggestion", "correction", "response", "addressed"],
    "FM-2.2": ["clear", "explicit", "specific"],
    "FM-2.3": ["main task", "goal", "query", "relevant"],
}


def score_step_for_contradiction(step: Dict[str, Any], error_type: str) -> Tuple[int, List[str]]:
    text = normalize_text(step["content"])
    stage = normalize_text(step["stage"])

    score = 0
    hits: List[str] = []

    for kw in CONTRADICT_KEYWORDS.get(error_type, []):
        if kw in text:
            score += 2
            hits.append(kw)

    for kw in GENERIC_CONTRADICT_HINTS.get(error_type, []):
        if kw in text:
            score += 1
            hits.append(kw)

    if error_type in CONTRADICT_STAGE_BONUS and stage in CONTRADICT_STAGE_BONUS[error_type]:
        if score > 0:
            score += 1
            hits.append(f"stage:{stage}")

    return score, hits


def score_explicit_contradiction(
    steps: List[Dict[str, Any]],
    error_type: str,
    max_steps: int = 4,
) -> Tuple[int, List[int], List[str]]:
    """
    Returns:
    - contradiction_score
    - evidence_steps
    - hit keywords
    """
    if not steps:
        return 0, [], []

    scored = []
    for s in steps:
        step_score, hits = score_step_for_contradiction(s, error_type)
        if step_score > 0:
            scored.append((step_score, s["step"], hits))

    if not scored:
        return 0, [], []

    scored.sort(key=lambda x: (-x[0], x[1]))

    top_items = scored[:max_steps]
    total_score = sum(x[0] for x in top_items)
    evidence_steps = sorted([x[1] for x in top_items])

    hit_words: List[str] = []
    for _, _, hits in top_items:
        hit_words.extend(hits)

    uniq_hits = []
    seen = set()
    for h in hit_words:
        if h not in seen:
            uniq_hits.append(h)
            seen.add(h)

    return total_score, evidence_steps, uniq_hits


def choose_contradict_errors(
    true_error: str,
    positive_errors: Set[str],
    steps: List[Dict[str, Any]],
    k: int = 1,
    min_score: int = 3,
) -> List[Dict[str, Any]]:
    """
    Prefer absent errors with explicit contradiction evidence.
    """
    nearby_map = {
        "FM-1.1": ["FM-2.3", "FM-1.4", "FM-1.2"],
        "FM-1.2": ["FM-1.1", "FM-2.3", "FM-2.5"],
        "FM-1.3": ["FM-2.1", "FM-3.2", "FM-2.3"],
        "FM-1.4": ["FM-2.5", "FM-2.6", "FM-1.1"],
        "FM-1.5": ["FM-3.1", "FM-3.2", "FM-3.3"],

        "FM-2.1": ["FM-1.3", "FM-2.3", "FM-2.5"],
        "FM-2.2": ["FM-2.4", "FM-2.3", "FM-2.5"],
        "FM-2.3": ["FM-1.1", "FM-2.2", "FM-2.1"],
        "FM-2.4": ["FM-2.2", "FM-2.5", "FM-1.4"],
        "FM-2.5": ["FM-1.4", "FM-2.4", "FM-2.6"],
        "FM-2.6": ["FM-1.4", "FM-2.5", "FM-3.3"],

        "FM-3.1": ["FM-1.5", "FM-3.2", "FM-3.3"],
        "FM-3.2": ["FM-3.3", "FM-1.5", "FM-3.1"],
        "FM-3.3": ["FM-3.2", "FM-2.6", "FM-3.1"],
    }
    # nearby_map = {
    #     # FM-1.x -> only category 2/3
    #     "FM-1.1": ["FM-2.2", "FM-2.4", "FM-3.2"],
    #     "FM-1.2": ["FM-2.2", "FM-2.4", "FM-3.2"],
    #     "FM-1.3": ["FM-2.4", "FM-3.1", "FM-3.2"],
    #     "FM-1.4": ["FM-2.4", "FM-2.5", "FM-3.3"],
    #     "FM-1.5": ["FM-2.3", "FM-3.1", "FM-3.2"],
    #
    #     # FM-2.x -> only category 1/3
    #     "FM-2.1": ["FM-1.3", "FM-1.5", "FM-3.1"],
    #     "FM-2.2": ["FM-1.1", "FM-1.2", "FM-3.2"],
    #     "FM-2.3": ["FM-1.1", "FM-1.2", "FM-3.1"],
    #     "FM-2.4": ["FM-1.4", "FM-1.1", "FM-3.3"],
    #     "FM-2.5": ["FM-1.4", "FM-1.1", "FM-3.3"],
    #     "FM-2.6": ["FM-1.4", "FM-1.1", "FM-3.3"],
    #
    #     # FM-3.x -> only category 1/2
    #     "FM-3.1": ["FM-1.5", "FM-2.3", "FM-2.4"],
    #     "FM-3.2": ["FM-1.3", "FM-2.3", "FM-2.4"],
    #     "FM-3.3": ["FM-1.4", "FM-2.5", "FM-2.6"],
    # }

    nearby_set = set(nearby_map.get(true_error, []))

    candidates = []
    for code in ALL_ERROR_CODES:
        if code == true_error:
            continue
        if code in positive_errors:
            continue

        score, evidence_steps, hit_words = score_explicit_contradiction(steps, code)
        if score >= min_score:
            nearby_bonus = 1 if code in nearby_set else 0
            candidates.append(
                (
                    nearby_bonus,
                    score,
                    code,
                    evidence_steps,
                    hit_words,
                )
            )

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    chosen = []
    used = set()
    for nearby_bonus, score, code, evidence_steps, hit_words in candidates:
        if code in used:
            continue
        chosen.append(
            {
                "error_type": code,
                "negative_type": "explicit_contradiction",
                "contradiction_score": score,
                "evidence_steps": evidence_steps,
                "contradiction_hits": hit_words,
            }
        )
        used.add(code)
        if len(chosen) >= k:
            break

    return chosen


def choose_neutral_errors(
    true_error: str,
    positive_errors: Set[str],
    excluded_errors: Optional[Set[str]] = None,
    k: int = 2,
) -> List[Dict[str, str]]:
    """
    Select absent errors as neutral negatives.
    Prefer nearby / hard negatives first.
    """
    if excluded_errors is None:
        excluded_errors = set()

    nearby_map = {
        "FM-1.1": ["FM-2.3"],
        "FM-1.2": ["FM-1.1"],
        "FM-1.3": ["FM-2.1"],
        "FM-1.4": ["FM-2.5"],
        "FM-1.5": ["FM-3.1"],

        "FM-2.1": ["FM-1.3"],
        "FM-2.2": ["FM-2.4"],
        "FM-2.3": ["FM-1.1"],
        "FM-2.4": ["FM-2.2"],
        "FM-2.5": ["FM-1.4"],
        "FM-2.6": ["FM-1.4"],

        "FM-3.1": ["FM-1.5"],
        "FM-3.2": ["FM-3.3"],
        "FM-3.3": ["FM-3.2"],
    }

    selected: List[Dict[str, str]] = []
    used = set(excluded_errors)

    for code in nearby_map.get(true_error, []):
        if code != true_error and code not in positive_errors and code not in used:
            selected.append({"error_type": code, "negative_type": "adjacent_absent_error"})
            used.add(code)
            if len(selected) >= k:
                return selected

    for code in ALL_ERROR_CODES:
        if code != true_error and code not in positive_errors and code not in used:
            selected.append({"error_type": code, "negative_type": "random_absent_error"})
            used.add(code)
            if len(selected) >= k:
                return selected

    return selected


# =========================================================
# 7. Main conversion (trajectory-level NLI / joint supervision)
# =========================================================

def convert_maserror_sample_to_nli(
    sample: Dict[str, Any],
    negatives_per_positive: int = 3,
    contradict_per_positive: int = 1,
    contradict_min_score: int = 3,
    include_query: bool = True,
    include_evidence: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert one maserror/agent_detection sample into flat NLI records.

    labels:
    - entail
    - neutral
    - contradict

    joint supervision:
    - entail    -> target_agent_names = all gold agents responsible for this error_type
    - neutral   -> target_agent_names = []
    - contradict-> target_agent_names = []

    Important:
    This version creates one positive record per positive error_type, not one
    positive record per (error_type, agent_name) pair. Therefore, if the same
    hypothesis is caused by multiple agents, the SFT response can contain
    multiple JSON objects, one per agent.
    """
    prompt_text = get_prompt_text(sample)
    query_text = extract_section(prompt_text, "QUERY:", "CONVERSATION HISTORY:")
    history_text = extract_section(prompt_text, "CONVERSATION HISTORY:", "## YOUR ANALYSIS:")

    steps = parse_conversation_history(history_text)
    candidate_agents = extract_agents_from_steps(steps)
    positive_by_error = extract_positive_faults_by_error(sample)

    trajectory_id = sample.get("id") or sample.get("data_source", "unknown_sample")
    premise = build_premise(query_text, history_text, include_query=include_query)

    records: List[Dict[str, Any]] = []
    rid = 0

    positive_error_set = set(positive_by_error.keys())

    for error_type, agent_names in positive_by_error.items():
        # Add all gold agents for this error type into candidate list if missing.
        local_candidate_agents = list(candidate_agents)
        for agent_name in agent_names:
            if agent_name not in local_candidate_agents:
                local_candidate_agents.append(agent_name)

        evidence_steps = select_evidence_steps(steps, error_type) if include_evidence else []

        # 1) positive: one record per error_type, multiple responsible agents allowed.
        rid += 1
        pos_item = {
            "id": f"{trajectory_id}_nli_{rid}",
            "trajectory_id": trajectory_id,
            "premise": premise,
            "hypothesis": build_hypothesis(error_type),
            "label": "entail",
            "target_error_type": error_type,
            "target_agent_names": agent_names,
            # Keep old field for backward compatibility/debugging; do not use it for SFT response.
            "target_agent_name": agent_names[0] if agent_names else None,
            "candidate_agents": local_candidate_agents,
            "is_positive": True,
            "negative_type": None,
            "anchor_error_type": error_type,
            "anchor_agent_names": agent_names,
            "anchor_agent_name": agent_names[0] if agent_names else None,
        }
        if include_evidence:
            pos_item["evidence_steps"] = evidence_steps
        records.append(pos_item)

        # 2) contradict negatives
        contradict_candidates = choose_contradict_errors(
            true_error=error_type,
            positive_errors=positive_error_set,
            steps=steps,
            k=min(contradict_per_positive, negatives_per_positive),
            min_score=contradict_min_score,
        )

        used_neg_errors = set()

        for neg in contradict_candidates:
            neg_error = neg["error_type"]
            used_neg_errors.add(neg_error)

            rid += 1
            neg_item = {
                "id": f"{trajectory_id}_nli_{rid}",
                "trajectory_id": trajectory_id,
                "premise": premise,
                "hypothesis": build_hypothesis(neg_error),
                "label": "contradict",
                "target_error_type": neg_error,
                "target_agent_names": [],
                "target_agent_name": None,
                "candidate_agents": local_candidate_agents,
                "is_positive": False,
                "negative_type": neg["negative_type"],
                "contradiction_score": neg["contradiction_score"],
                "contradiction_hits": neg["contradiction_hits"],
                "anchor_error_type": error_type,
                "anchor_agent_names": agent_names,
                "anchor_agent_name": agent_names[0] if agent_names else None,
            }
            if include_evidence:
                neg_item["evidence_steps"] = neg["evidence_steps"]
            records.append(neg_item)

        # 3) neutral negatives
        remain_neg = max(0, negatives_per_positive - len(contradict_candidates))
        if remain_neg > 0:
            neutral_candidates = choose_neutral_errors(
                true_error=error_type,
                positive_errors=positive_error_set,
                excluded_errors=used_neg_errors,
                k=remain_neg,
            )

            for neg in neutral_candidates:
                neg_error = neg["error_type"]
                neg_type = neg["negative_type"]
                neg_evidence_steps = select_evidence_steps(steps, neg_error) if include_evidence else []

                rid += 1
                neg_item = {
                    "id": f"{trajectory_id}_nli_{rid}",
                    "trajectory_id": trajectory_id,
                    "premise": premise,
                    "hypothesis": build_hypothesis(neg_error),
                    "label": "neutral",
                    "target_error_type": neg_error,
                    "target_agent_names": [],
                    "target_agent_name": None,
                    "candidate_agents": local_candidate_agents,
                    "is_positive": False,
                    "negative_type": neg_type,
                    "anchor_error_type": error_type,
                    "anchor_agent_names": agent_names,
                    "anchor_agent_name": agent_names[0] if agent_names else None,
                }
                if include_evidence:
                    neg_item["evidence_steps"] = neg_evidence_steps
                records.append(neg_item)

    return records


# =========================================================
# 8. Optional group format
# =========================================================

def convert_maserror_sample_to_nli_groups(
    sample: Dict[str, Any],
    negatives_per_positive: int = 3,
    contradict_per_positive: int = 1,
    contradict_min_score: int = 3,
    include_query: bool = True,
    include_evidence: bool = False,
) -> Dict[str, Any]:
    flat_records = convert_maserror_sample_to_nli(
        sample,
        negatives_per_positive=negatives_per_positive,
        contradict_per_positive=contradict_per_positive,
        contradict_min_score=contradict_min_score,
        include_query=include_query,
        include_evidence=include_evidence,
    )

    groups: Dict[str, Dict[str, Any]] = {}

    for rec in flat_records:
        # Group by anchor error type. One anchor error may correspond to multiple responsible agents.
        group_key = f"{rec.get('anchor_error_type')}"
        if group_key not in groups:
            groups[group_key] = {
                "group_id": group_key,
                "trajectory_id": rec["trajectory_id"],
                "anchor_error_type": rec.get("anchor_error_type"),
                "anchor_agent_names": rec.get("anchor_agent_names", []),
                "samples": [],
            }
        groups[group_key]["samples"].append(rec)

    return {
        "trajectory_id": sample.get("id") or sample.get("data_source", "unknown_sample"),
        "nli_groups": list(groups.values()),
    }


# =========================================================
# 9. Convert NLI -> SFT verifier data
# =========================================================

def label_to_abc(label: str) -> str:
    mapping = {
        "entail": "A",
        "neutral": "B",
        "contradict": "C",
    }
    if label not in mapping:
        raise ValueError(f"Unsupported label: {label}")
    return mapping[label]


def build_joint_json_response(label: str, agent_names: Optional[List[str]] = None) -> str:
    """
    Build SFT response in multi-JSON-object format.

    For A:
        {"label":"A","agents":["Planner"]}
        {"label":"A","agents":["Solver"]}

    For B/C:
        {"label":"B","agents":[]}
        {"label":"C","agents":[]}
    """
    label_token = label_to_abc(label)
    agent_names = agent_names or []

    if label_token == "A":
        if not agent_names:
            # This case should be rare for training data, but it is allowed by the prompt.
            return json.dumps({"label": label_token, "agents": []}, ensure_ascii=False)

        lines = []
        seen: Set[str] = set()
        for agent_name in agent_names:
            if not agent_name or agent_name in seen:
                continue
            seen.add(agent_name)
            obj = {
                "label": label_token,
                "agents": [agent_name],
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        return "\n".join(lines)

    obj = {
        "label": label_token,
        "agents": [],
    }
    return json.dumps(obj, ensure_ascii=False)


def build_sft_prompt(rec: Dict[str, Any]) -> str:
    candidate_agents = rec.get("candidate_agents", [])
    if candidate_agents:
        candidate_agents_text = "\n".join([f"- {a}" for a in candidate_agents])
    else:
        candidate_agents_text = "- None"

    prompt = (
        STRICT_VERIFIER_PROMPT
        + "\n\nTrajectory:\n"
        + rec["premise"]
        + "\n\nHypothesis:\n"
        + rec["hypothesis"]
        + "\n\nCandidate agents:\n"
        + candidate_agents_text
    )
    return prompt


def nli_records_to_sft(
    nli_records: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    sft_records = []

    for rec in nli_records:
        prompt = build_sft_prompt(rec)
        response = build_joint_json_response(
            label=rec["label"],
            agent_names=rec.get("target_agent_names", []),
        )

        sft_records.append(
            {
                "id": rec["id"],
                "prompt": prompt,
                "response": response,
            }
        )

    return sft_records


# =========================================================
# 10. JSONL helpers
# =========================================================

def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# =========================================================
# 11. Batch conversion over a whole file
# =========================================================

def convert_jsonl_file(
    input_path: str,
    nli_out_path: str,
    sft_out_path: Optional[str] = None,
    negatives_per_positive: int = 3,
    contradict_per_positive: int = 1,
    contradict_min_score: int = 3,
    include_query: bool = True,
    include_evidence_in_nli: bool = False,
) -> None:
    all_nli: List[Dict[str, Any]] = []
    all_sft: List[Dict[str, Any]] = []

    label_counter = {
        "entail": 0,
        "neutral": 0,
        "contradict": 0,
    }

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] Failed to parse line {line_idx}: {e}")
                continue

            try:
                nli_records = convert_maserror_sample_to_nli(
                    sample,
                    negatives_per_positive=negatives_per_positive,
                    contradict_per_positive=contradict_per_positive,
                    contradict_min_score=contradict_min_score,
                    include_query=include_query,
                    include_evidence=include_evidence_in_nli,
                )
                all_nli.extend(nli_records)

                for r in nli_records:
                    label_counter[r["label"]] += 1

                if sft_out_path is not None:
                    sft_records = nli_records_to_sft(nli_records)
                    all_sft.extend(sft_records)

            except Exception as e:
                sample_id = sample.get("id", f"line_{line_idx}")
                print(f"[Warning] Failed to convert sample {sample_id}: {e}")

    save_jsonl(all_nli, nli_out_path)
    print(f"[Done] Saved NLI records to: {nli_out_path}")

    if sft_out_path is not None:
        save_jsonl(all_sft, sft_out_path)
        print(f"[Done] Saved SFT records to: {sft_out_path}")

    print(
        "[Stats] label counts:",
        label_counter,
        "total=",
        sum(label_counter.values()),
    )


# =========================================================
# 12. Example usage
# =========================================================

if __name__ == "__main__":
    # Example single-sample usage
    sample = {
        "id": "demo_sample_1",
        "data_source": "maserror/agent_detection",
        "prompt": [
            {
                "role": "user",
                "content": """## CONVERSATION TO ANALYZE:
QUERY:
Write a Python function for date validation.

CONVERSATION HISTORY:
Step 1 - RoleAssigner () [initialization]:
Assigned agents and clearly stated the task and evaluation plan.

Step 2 - Solver () [reasoning]:
Produced a solution in Python as requested and considered the previous correction from the user.

Step 3 - Evaluator () [evaluation]:
Validated the answer with test cases and checked the expected outputs before finalizing.

## YOUR ANALYSIS:
"""
            }
        ],
        "reward_model": {
            "ground_truth": """{
  "faulty_agents": [
    {"agent_name": "Solver", "error_type": "FM-1.4"}
  ]
}"""
        }
    }

    # Single-sample debug
    # nli_records = convert_maserror_sample_to_nli(
    #     sample,
    #     negatives_per_positive=2,
    #     contradict_per_positive=1,
    #     contradict_min_score=3,
    #     include_query=True,
    #     include_evidence=True,
    # )
    # print(json.dumps(nli_records, indent=2, ensure_ascii=False))
    #
    # sft_records = nli_records_to_sft(nli_records)
    # print(json.dumps(sft_records[:5], indent=2, ensure_ascii=False))

    # File-level conversion
    input_path = "data/train_aegis_verl.jsonl"
    nli_out_path = "data/train_aegis_nli_no_agent_multi_agent_jsonl_8B.jsonl"
    sft_out_path = "data/train_aegis_sft_no_agent_multi_agent_jsonl_8B.jsonl"

    convert_jsonl_file(
        input_path=input_path,
        nli_out_path=nli_out_path,
        sft_out_path=sft_out_path,
        negatives_per_positive=2,
        contradict_per_positive=1,
        contradict_min_score=3,
        include_query=True,
        include_evidence_in_nli=False,
    )