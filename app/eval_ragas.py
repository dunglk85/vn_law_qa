import argparse
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
from ragas.testset import TestsetGenerator

from app.config import config
from app.core.ingest_service import _load_docs


BASE_DIR = Path(__file__).parent.parent
TEST_DATA_PATH = BASE_DIR / "seed" / "qna_test.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"
API_URL = "http://localhost:8000/ask"
REQUEST_TIMEOUT = 30.0
DEFAULT_TEST_SIZE = 20


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


def generate_testset(
    output_path: Path = TEST_DATA_PATH,
    test_size: int = DEFAULT_TEST_SIZE,
) -> None:
    """Generate test Q&A pairs from source documents using RAGAS TestsetGenerator."""
    print(f"📄 Loading documents from {config.data_dir}...")
    docs = asyncio.run(_load_docs())
    print(f"📄 Loaded {len(docs)} documents")

    if not docs:
        print("❌ No documents found. Check DATA_DIR in .env")
        return

    generator_llm = ChatOpenAI(model=config.llm_model)
    critic_llm = ChatOpenAI(model=config.llm_model)

    generator = TestsetGenerator.with_openai(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
    )

    print(f"🧪 Generating {test_size} test samples...")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        test_size=test_size,
    )

    df = testset.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            entry = {
                "question": row.get("question", ""),
                "answer": row.get("ground_truth", ""),
            }
            if entry["question"] and entry["answer"]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"💾 Generated {len(df)} test samples → {output_path}")


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
    parser = argparse.ArgumentParser(description="RAGAS evaluation harness")
    parser.add_argument(
        "mode",
        choices=["generate", "evaluate"],
        help="generate: create test dataset from documents; evaluate: run evaluation against live server",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f"Number of test samples to generate (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_DATA_PATH,
        help=f"Path to test data JSONL file (default: {TEST_DATA_PATH})",
    )
    args = parser.parse_args()

    if args.mode == "generate":
        generate_testset(output_path=args.test_path, test_size=args.test_size)
    else:
        asyncio.run(evaluate_rag_system(test_path=args.test_path))
