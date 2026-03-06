import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
import uuid

# Ensure project root is importable when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.integrated_pipeline import IntegratedPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipeline until pre-Sora prompt generation and print the prompt."
    )
    parser.add_argument(
        "--image-url",
        required=True,
        help="Image URL to process",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Optional task id. If omitted, generated automatically.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result in JSON format",
    )
    return parser.parse_args()


async def run_pre_sora_prompt(image_url: str, task_id: str) -> dict:
    pipeline = IntegratedPipeline(task_id)
    pipeline.source_image_url = image_url

    extracted_text = await pipeline._step_ocr(image_url)
    sentences = await pipeline._step_segmentation(extracted_text)
    integrated_result = await pipeline._step_process_sentences(sentences)

    return {
        "task_id": task_id,
        "image_url": image_url,
        "ocr_text": extracted_text,
        "sentence_count": len(sentences),
        "sign_data_count": len(integrated_result.get("sign_data", [])),
        "image_context": pipeline.image_context,
        "character_description": pipeline.character_description,
        "video_prompt": integrated_result.get("video_prompt", ""),
    }


def build_task_id(raw_task_id: str | None) -> str:
    if raw_task_id:
        return raw_task_id
    return f"pre_sora_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


def print_text_result(result: dict) -> None:
    print("=== PRE-SORA PROMPT TEST ===")
    print("TASK_ID:", result["task_id"])
    print("IMAGE_URL:", result["image_url"])
    print("OCR_TEXT:", result["ocr_text"])
    print("SENTENCE_COUNT:", result["sentence_count"])
    print("SIGN_DATA_COUNT:", result["sign_data_count"])
    print("CHARACTER_DESCRIPTION:", result["character_description"])
    print("IMAGE_CONTEXT:", result["image_context"])
    print("PROMPT_LEN:", len(result["video_prompt"]))
    print("=== PROMPT START ===")
    print(result["video_prompt"])
    print("=== PROMPT END ===")


def main() -> None:
    args = parse_args()
    task_id = build_task_id(args.task_id)
    result = asyncio.run(run_pre_sora_prompt(args.image_url, task_id))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text_result(result)


if __name__ == "__main__":
    main()
