import argparse
import json
import re
from typing import List, Tuple
from openai import OpenAI

# 模型配置
api_key = "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"
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-chat-v3-0324:free"

def parse_instructions(json_path: str) -> List[str]:
    """
    从 input.json 中提取 special_instructions 字段，支持按编号(1. 2. 3.)分隔的格式。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("special_instructions", "")

        # 正则提取编号分隔的约束项（如 1. ... 2. ...）
        pattern = r"\d+\.\s+(.*?)(?=(\d+\.|$))"
        matches = re.findall(pattern, raw, flags=re.DOTALL)

        # 提取内容部分并清理空格
        instructions = [item[0].strip() for item in matches]
        return instructions
    except Exception as e:
        raise RuntimeError(f"无法读取或解析 {json_path}: {e}")

def read_code(code_path: str) -> str:
    """
    读取 Python 源代码文件
    """
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"无法读取代码文件 {code_path}: {e}")

def call_llm(prompt: str) -> str:
    """
    调用 LLM 接口并返回响应内容
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def check_constraints_with_llm(code_content: str, constraints: List[str]) -> List[bool]:
    """
    使用 LLM 判断代码是否满足每一条约束，返回布尔结果列表
    """
    results = []
    for idx, constraint in enumerate(constraints, start=1):
        prompt = f"""You are a code review expert responsible for determining whether a piece of model training code complies with specific modeling constraints.\\
Please evaluate the following constraint based on the code below:

- Constraint {idx}: {constraint}

Code:
{code_content}

Your response must follow this format exactly:
Answer: [Yes/No]
Reason: [Brief and specific justification explaining why the code complies with the constraint or lacks sufficient information to verify compliance]\\
Important:
If the code does not include relevant evidence, conservatively answer "No".\\
Do not include any additional explanations, comments, or formatting beyond the required two lines.
"""
        # print(prompt)
        response = call_llm(prompt)
        answer_line = response.strip().splitlines()[0].lower().replace("：", ":")
        results.append("yes" in answer_line)
    return results

def score_constraints(results: List[bool]) -> Tuple[int, int, float]:
    """
    统计约束通过数和通过率
    """
    total = len(results)
    passed = sum(results)
    rate = passed / total if total > 0 else 0.0
    return passed, total, rate

def main():
    parser = argparse.ArgumentParser(description="评估模型代码是否满足建模约束")
    parser.add_argument("--input", type=str, required=True, help="input.json 文件路径")
    parser.add_argument("--code", type=str, required=True, help="代码文件路径(如 best_solution.py)")
    args = parser.parse_args()

    instructions = parse_instructions(args.input)
    # print(instructions)
    code_content = read_code(args.code)
    # print(code_content)
    results = check_constraints_with_llm(code_content, instructions)
    passed, total, rate = score_constraints(results)

    print("评估结果：\n")
    for idx, (inst, passed_item) in enumerate(zip(instructions, results), start=1):
        status = "✔️ 满足" if passed_item else "❌ 不满足"
        print(f"{idx}. {inst} → {status}")
    print(f"\n总通过率: {passed}/{total} = {rate:.2%}")

if __name__ == "__main__":
    main()
