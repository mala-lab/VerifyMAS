# Copyright 2024 Your Name/Organization
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

import json
from typing import Set, Tuple, Optional
import re

def extract_attributions(solution_str: str) -> Optional[Set[Tuple[str, str]]]:
    """
    从模型输出或标准答案的字符串中解析出错误归因对。
    支持从CoT格式的输出中提取JSON。
    
    Args:
        solution_str: 包含 "faulty_agents" 列表的JSON格式字符串，或CoT分析后的JSON。

    Returns:
        一个包含 (agent_name, error_type) 元组的集合，如果解析失败则返回 None。
    """
    try:
        original_str = solution_str.strip()
        
        # 方法1: 直接尝试解析整个字符串（适用于纯JSON输出）
        try:
            data = json.loads(original_str)
            if "faulty_agents" in data:
                faulty_agents = data.get("faulty_agents", [])
                attribution_set = set()
                for agent_info in faulty_agents:
                    agent_name = agent_info.get("agent_name").replace(" ", "").lower()
                    error_type = agent_info.get("error_type").replace(" ", "").lower()
                    if agent_name and error_type:
                        attribution_set.add((agent_name, error_type))
                return attribution_set
        except json.JSONDecodeError:
            pass
        
        # 方法2: 移除Markdown代码块标记后尝试解析
        if original_str.startswith("```json"):
            json_str = original_str[7:-3].strip()
        elif original_str.startswith("```"):
            json_str = original_str[3:-3].strip()
        else:
            json_str = original_str
            
        try:
            data = json.loads(json_str)
            if "faulty_agents" in data:
                faulty_agents = data.get("faulty_agents", [])
                attribution_set = set()
                for agent_info in faulty_agents:
                    agent_name = agent_info.get("agent_name").replace(" ", "").lower()
                    error_type = agent_info.get("error_type").replace(" ", "").lower()
                    if agent_name and error_type:
                        attribution_set.add((agent_name, error_type))
                return attribution_set
        except json.JSONDecodeError:
            pass
        
        # 方法3: 从CoT输出中查找JSON块（查找包含faulty_agents的JSON）
        # 使用正则表达式查找所有可能的JSON对象
        json_pattern = r'\{[^{}]*"faulty_agents"[^{}]*\}'
        matches = re.findall(json_pattern, original_str, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if "faulty_agents" in data:
                    faulty_agents = data.get("faulty_agents", [])
                    attribution_set = set()
                    for agent_info in faulty_agents:
                        agent_name = agent_info.get("agent_name").replace(" ", "").lower()
                        error_type = agent_info.get("error_type").replace(" ", "").lower()
                        if agent_name and error_type:
                            attribution_set.add((agent_name, error_type))
                    return attribution_set
            except json.JSONDecodeError:
                continue
        
        # 方法4: 更宽泛的JSON查找（查找任何包含大括号的内容）
        json_blocks = re.findall(r'\{.*?\}', original_str, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if "faulty_agents" in data:
                    faulty_agents = data.get("faulty_agents", [])
                    attribution_set = set()
                    for agent_info in faulty_agents:
                        agent_name = agent_info.get("agent_name").replace(" ", "").lower()
                        error_type = agent_info.get("error_type").replace(" ", "").lower()
                        if agent_name and error_type:
                            attribution_set.add((agent_name, error_type))
                    return attribution_set
            except json.JSONDecodeError:
                continue
        
        # 方法5: 更宽松的JSON查找 - 查找包含faulty_agents的任何内容
        # 尝试从文本中提取JSON-like结构
        faulty_agents_pattern = r'faulty_agents["\s]*:["\s]*\[(.*?)\]'
        matches = re.findall(faulty_agents_pattern, original_str, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                # 尝试解析数组内容
                # 先尝试直接解析
                try:
                    agents_data = json.loads(f"[{match}]")
                except json.JSONDecodeError:
                    # 如果失败，尝试修复常见的格式问题
                    # 替换单引号为双引号
                    fixed_match = match.replace("'", '"')
                    # 移除多余的逗号
                    fixed_match = re.sub(r',\s*}', '}', fixed_match)
                    fixed_match = re.sub(r',\s*]', ']', fixed_match)
                    try:
                        agents_data = json.loads(f"[{fixed_match}]")
                    except json.JSONDecodeError:
                        continue
                
                attribution_set = set()
                for agent_info in agents_data:
                    if isinstance(agent_info, dict):
                        agent_name = agent_info.get("agent_name", "").replace(" ", "").lower()
                        error_type = agent_info.get("error_type", "").replace(" ", "").lower()
                        if agent_name and error_type:
                            attribution_set.add((agent_name, error_type))
                
                if attribution_set:
                    return attribution_set
            except (json.JSONDecodeError, TypeError):
                continue
                
        return None
    except (TypeError, AttributeError):
        return None

def compute_score(
    solution_str: str, 
    ground_truth: str, 
    pair_credit: float = 1.0,
    agent_credit: float = 0.3,
    error_type_credit: float = 0.1,
    fp_penalty: float = 0.2,
    malformed_penalty: float = -1.0,
    duplicate_penalty: float = 0.3,
    parseable_bonus: float = 0.1
) -> float:
    """
    为单次多智能体错误归因预测计算一个分层的奖励分数。
    这个函数是为强化学习设计的，它计算单次预测的得分，而不是批次的F1分数。

    Args:
        prediction_str: 模型生成的JSON格式字符串。
        ground_truth_str: 标准答案的JSON格式字符串。
        pair_credit: 完全匹配一个 (agent, error_type) 对获得的奖励。
        agent_credit: 只匹配了 agent_name 时获得的"部分"奖励。
        error_type_credit: 只匹配了 error_type 时获得的"部分"奖励。
        fp_penalty: 对于每一个错误的预测（False Positive）应用的惩罚。
        malformed_penalty: 如果模型输出格式错误，给予的惩罚分数。
        duplicate_penalty: 对重复预测同一agent的惩罚。
        parseable_bonus: 如果能成功解析JSON格式，给予的小额奖励（鼓励正确格式）。

    Returns:
        最终的标量奖励分数。
    """
    # print(f"[DEBUG] solution_str: {solution_str}")
    # print(f"[DEBUG] ground_truth: {ground_truth}")
    pred_attributions = extract_attributions(solution_str)
    true_attributions = extract_attributions(ground_truth)
    # print(f"[DEBUG] pred_attributions: {pred_attributions}")
    # print(f"[DEBUG] true_attributions: {true_attributions}")

    # 如果模型输出格式错误或无法解析，直接返回惩罚分数
    if pred_attributions is None:
        return malformed_penalty
    
    # 基础分数：如果能成功解析JSON，给予小额奖励
    achieved_score = parseable_bonus
    
    # 如果标准答案为空（不应发生，但作为保护），且预测也为空，则满分
    if not true_attributions:
        return pair_credit + parseable_bonus if not pred_attributions else parseable_bonus

    # 提取 Agent 和 Error Type 的集合，用于部分分计算
    true_agents = {agent for agent, error in true_attributions}
    true_errors = {error for agent, error in true_attributions}
    
    # 跟踪已经获得部分奖励的agents和error_types，防止重复exploit
    rewarded_agents = set()
    rewarded_error_types = set()
    pred_agents = [agent for agent, _ in pred_attributions]
    pred_errors = [error for _, error in pred_attributions]
    
    # 对重复预测同一agent的行为进行惩罚
    agent_counts = {}
    for agent in pred_agents:
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
    
    # 对重复预测同一error_type的行为进行惩罚
    error_counts = {}
    for error in pred_errors:
        error_counts[error] = error_counts.get(error, 0) + 1
    
    duplicate_penalty_total = 0.0
    for agent, count in agent_counts.items():
        if count > 1:
            duplicate_penalty_total += (count - 1) * duplicate_penalty
    
    for error, count in error_counts.items():
        if count > 1:
            duplicate_penalty_total += (count - 1) * duplicate_penalty * 0.5  # error重复惩罚稍轻
    
    achieved_score -= duplicate_penalty_total
    
    # 遍历模型的每一个预测，进行计分
    for pred_pair in pred_attributions:
        pred_agent, pred_error = pred_pair
        
        # 1. Pair Level (最高奖励): 检查 (agent, error) 对是否完全匹配
        if pred_pair in true_attributions:
            achieved_score += pair_credit
        # 2. Agent Level (部分奖励): 每个正确agent只能获得一次奖励
        elif pred_agent in true_agents and pred_agent not in rewarded_agents:
            achieved_score += agent_credit
            rewarded_agents.add(pred_agent)
        # 3. Error Type Level (最低奖励): 每个正确error_type只能获得一次奖励，且agent必须错误
        elif (pred_agent not in true_agents and 
              pred_error in true_errors and 
              pred_error not in rewarded_error_types):
            achieved_score += error_type_credit
            rewarded_error_types.add(pred_error)
        # 4. False Positive 惩罚: 如果预测完全错误
        else:
            achieved_score -= fp_penalty

    # 计算理论上的最高可能分数，考虑parseable_bonus
    max_possible_score = (len(true_attributions) * pair_credit if true_attributions else pair_credit) + parseable_bonus
    
    # 添加"数量攻击"防护：如果预测数量过多（超过真实答案2倍），额外惩罚
    if len(pred_attributions) > len(true_attributions) * 2:
        quantity_penalty = (len(pred_attributions) - len(true_attributions) * 2) * fp_penalty * 0.5
        achieved_score -= quantity_penalty

    # 归一化奖励，使其在 [-1, 1] 区间附近，为RL提供稳定信号
    # 注意：如果FP很多，分数可能为负
    # parseable_bonus提供了一个基础分数，鼓励格式正确的输出
    # print(f"[DEBUG] achieved_score: {achieved_score}")
    # print(f"[DEBUG] max_possible_score: {max_possible_score}")
    return achieved_score / max_possible_score if max_possible_score > 0 else parseable_bonus