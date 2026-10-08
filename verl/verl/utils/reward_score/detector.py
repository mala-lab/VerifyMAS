# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def extract_solution(solution_str, method="strict"):
    assert method == "strict"
    lines = solution_str.strip().splitlines()
    answer = None
    for i in range(len(lines)-1, -1, -1):
        line = lines[i]
        match = re.match(r"SUSPECTED[_ ]ROLE:\s*(\S+)", line)
        if match:
            answer = match.group(1).strip()
            break
        # 如果这一行是 SUSPECTED ROLE: 或 SUSPECTED_ROLE:，下一行是答案
        if re.match(r"SUSPECTED[_ ]ROLE:\s*$", line) and i+1 < len(lines):
            answer = lines[i+1].strip()
            break
    return answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, **kwargs):
    """The scoring function for detector.

    Args:
        solution_str: the solution text (should contain 'SUSPECTED_ROLE: XXX' at the end)
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    print(f"======================\n solution_str: {solution_str} , answer: {answer} , ground_truth: {ground_truth}\n======================\n")
    if answer is None:
        return format_score
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score 