import json
import requests
import time
from typing import List, Dict, Any
import argparse
import logging
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add the project root to the Python path
# This allows us to import from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.countdown import extract_solution, compute_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLLMProcessor:
    def __init__(self, endpoint_url: str, model_name: str = None, max_tokens: int = 2048,
                 temperature: float = 0.7, request_timeout: int = 60):
        """
        Initialize the vLLM processor.

        Args:
            endpoint_url: The vLLM server endpoint (e.g., "http://localhost:8000/v1/completions")
            model_name: Model name (optional, can be specified in API call)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            request_timeout: Request timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

    def make_api_call(self, prompt: str) -> str:
        """
        Make a call to the vLLM endpoint.

        Args:
            prompt: The input prompt

        Returns:
            Generated response text
        """
        payload = {
        "messages":  [{
                "content": prompt,
                "role": "user"
            }],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "stop": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0,
            "repetition_penalty": 1.2
        }

        # Add model name if specified
        if self.model_name:
            payload["model"] = self.model_name

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            result = response.json()

            # Extract the generated text from vLLM response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ""

    def process_file(self, input_file: str, output_file: str = None,
                    delay_between_requests: float = 0.1, resume: bool = False, limit: int = None):
        """
        Process a JSONL file by making API calls for each prompt.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file (defaults to input_file if not specified)
            delay_between_requests: Delay between API calls in seconds
            resume: Whether to resume from where we left off (skip entries with non-empty responses)
            limit: Limit the number of prompts to process (for testing)
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = input_file

        # Read all entries
        entries = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON on line {line_num}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return

        logger.info(f"Loaded {len(entries)} entries from {input_file}")

        if limit is not None and limit < len(entries):
            logger.info(f"Limiting processing to the first {limit} entries.")
            entries = entries[:limit]

        # Process entries
        processed_count = 0
        skipped_count = 0
        total_score = 0.0
        num_entries = len(entries)

        with tqdm(total=num_entries, desc="Evaluating Leaderboard", unit="prompt") as pbar:
            for i, entry in enumerate(entries):
                if "num" not in entry or "target" not in entry:
                    # Silently skip malformed entries
                    continue

                # Skip if resuming and response already exists
                if resume and entry.get("response", "").strip():
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Construct the detailed prompt
                numbers_str = ", ".join(map(str, entry['num']))
                prompt = (
                    f"Using the numbers [{numbers_str}], create an equation that equals {entry['target']}. "
                    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                    "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
                    "for example <answer> (1 + 2) / 3 </answer>."
                )

                # Make API call
                raw_response = self.make_api_call(prompt)

                # Calculate score
                score = 0.0
                ground_truth = {"numbers": entry["num"], "target": entry["target"]}
                score = compute_score(solution_str=raw_response, ground_truth=ground_truth)
                total_score += score
                
                extracted_response = extract_solution(raw_response) if raw_response else ""
                
                if extracted_response:
                    processed_count += 1
                
                # Overwrite the entry with a clean dictionary containing only the desired fields
                entries[i] = {
                    "num": entry["num"],
                    "target": entry["target"],
                    "response": extracted_response or ""
                }

                # Update progress bar
                pbar.set_postfix({"avg_score": f"{total_score / (i + 1):.4f}"})
                pbar.update(1)

                # Add delay between requests to avoid overwhelming the server
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')

            logger.info(f"✓ Saved results to {output_file}")
            
            if num_entries > 0:
                final_avg_score = total_score / num_entries
                logger.info(f"\n=======================================================")
                logger.info(f"FINAL LEADERBOARD SCORE: {final_avg_score:.4f} ({total_score:.2f}/{num_entries})")
                logger.info(f"=======================================================")

            logger.info(f"Summary: Found <answer> tag in {processed_count} responses, Skipped {skipped_count} entries")

        except Exception as e:
            logger.error(f"Failed to save output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL file with vLLM API calls for leaderboard submission")
    parser.add_argument("--input_file", help="Input JSONL file path", default="leaderboard/submission_countdown.json")
    parser.add_argument("--output", "-o", help="Output JSONL file path (defaults to input file)", default="leaderboard/submission_countdown_results.json")
    parser.add_argument("--endpoint", "-e", default="http://localhost:8000/v1/chat/completions",
                       help="vLLM endpoint URL (e.g., http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--model", "-m", help="Model name (optional)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--delay", type=float, default=0,
                       help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing (skip entries with existing responses)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of prompts to process (for testing)")

    args = parser.parse_args()

    # Initialize processor
    processor = VLLMProcessor(
        endpoint_url=args.endpoint,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.timeout
    )

    # Process file
    try:
        processor.process_file(
            input_file=args.input_file,
            output_file=args.output,
            delay_between_requests=args.delay,
            resume=args.resume,
            limit=args.limit
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
