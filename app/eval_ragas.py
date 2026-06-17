import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd
from langchain_openai import ChatOpenAI
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig

from app.config import config


BASE_DIR = Path(__file__).parent.parent
TEST_DATA_PATH = BASE_DIR / "seed" / "qna_test.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"
API_URL = "http://localhost:8000/ask"
REQUEST_TIMEOUT = 30.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_eval_res(eval_result) -> None:
    scores = eval_result.scores
    if not scores:
        print("No scores to display.")
        return

    eval_str = ' | Q | '
    for k in scores[0].keys():
        eval_str = eval_str + str(k) + ' | '
    print(eval_str)

    for i, score in enumerate(scores):
        eval_str = ' | ' + str(i + 1) + ' | '
        for k in score.keys():
            eval_str = eval_str + str(score[k]) + ' | '
        print(eval_str)

    res = eval_result.to_pandas()
    means = res.mean(numeric_only=True).to_dict()
    print("\n📈 Averages:")
    for k, v in means.items():
        print(f"- {k}: {v:.3f}")


async def evaluate_rag_system(test_path: Path = TEST_DATA_PATH) -> None:
    test_data = load_jsonl(test_path)
    results = []

    oai_llm = ChatOpenAI(model=config.llm_model)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for item in test_data:
            question = item["question"]
            reference_answer = item["answer"]

            try:
                response = await client.post(API_URL, json={"question": question})
                response.raise_for_status()
                res = response.json()
                answer = res["answer"]
                contexts = res["contexts"]
            except httpx.HTTPError as e:
                print(f"⚠️  Failed to get answer for question: {question[:50]}... — {e}")
                continue

            results.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference_answer,
            ))

    if not results:
        print("❌ No successful evaluations. Check if the server is running.")
        return

    ds = EvaluationDataset(results)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(max_workers=8, timeout=60)
    eval_result = evaluate(dataset=ds, metrics=metrics, llm=oai_llm, run_config=run_config)

    print("RAGAS Evals Results")
    print_eval_res(eval_result)

    eval_result.to_pandas().to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"\n💾 Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(evaluate_rag_system())
