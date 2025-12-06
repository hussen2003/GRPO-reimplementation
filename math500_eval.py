import os
import re
import json
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATHS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # base model
    r"C:\###\###\###\deepseek-project\Qwen2-0.5B-Instruct-gsm8k-grpo-run1",  # GRPO run1 (change local ###)
    "Hpremier/Qwen-0.5B-Instruct-gsm8k-GRPO",  # final GRPO on HuggingFace 
]

N_PROBLEMS = 100  # how many MATH-500 questions to use (500 max)
DATASET_NAME = "HuggingFaceH4/MATH-500"

def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else {"": device},
    )
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 256) -> str:
    "Generate model's answers using sampling"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    # drop the prompt tokens from prompt
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


# answer cleaning / parsing
BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")

def clean_gold(ans: str) -> str:
    "Remove LaTeX"
    ans = ans.strip()
    m = BOXED_RE.fullmatch(ans)
    if m:
        ans = m.group(1)
    return ans.strip()


def extract_pred(text: str) -> str:
    "extract numeric answer, take last non-empty line, strip trailing period"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    last = lines[-1]
    if last.endswith("."):
        last = last[:-1]
    return last.strip()


# dataset 
def load_math500_subset(n_problems: int):
    print("\nLoading MATH-500 test data...")
    ds = load_dataset(DATASET_NAME, split="test")
    total = len(ds)
    print(f"Total test samples in split: {total}")

    if n_problems is None or n_problems >= total:
        return ds, total

    # deterministic subset
    subset = ds.select(range(n_problems))
    return subset, n_problems


def evaluate_model_on_math500(model_id: str, n_problems: int):
    tokenizer, model = load_model(model_id)
    problems, total = load_math500_subset(n_problems)

    correct = 0
    results = []

    print(f"Evaluating: {total} problems...")
    for i, ex in enumerate(problems, start=1):
        question = ex.get("problem") or ex.get("question")
        gold = ex.get("answer")
        if question is None or gold is None:
            continue

        gold_clean = clean_gold(str(gold))
        prompt = (
            question
            + "\n\nSolve and give the only the final numeric answer on the last line."
        )

        pred_full = generate_answer(tokenizer, model, prompt)
        pred_clean = extract_pred(pred_full)

        is_correct = (pred_clean == gold_clean)
        correct += int(is_correct)

        # per-problem debug
        print(f"\nProblem {i}")
        print(f"Gold: {gold_clean!r}")
        print(f"Model output: {pred_full[:120]!r}...")
        print(f"Parsed: {pred_clean!r}")
        print(f"Correct: {is_correct}")

        results.append(
            {
                "index": i,
                "gold": gold_clean,
                "prediction": pred_clean,
                "correct": bool(is_correct),
            }
        )

    #summary statistics
    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model_path": model_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_name = f"math500_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "per_problem": results},
            f,
            indent=2,
        )

    print("\nFinal Results")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("\n", summary)
    print(f"\nResults saved to {out_name}\n")

    return summary


if __name__ == "__main__":
    "evaluate all models in MODEL_PATHS"
    all_summaries = []
    for m in MODEL_PATHS:
        all_summaries.append(evaluate_model_on_math500(m, N_PROBLEMS))

    print("\n=== Overall summary across models ===")
    for s in all_summaries:
        print(
            f"{s['model_path']}: "
            f"accuracy={s['accuracy']*100:.2f}%  "
            f"({s['correct']}/{s['total']})"
        )