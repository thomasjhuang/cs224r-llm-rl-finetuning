import json
import requests
import time
from typing import List, Dict, Any
import argparse
import logging
from pathlib import Path

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
            "top_p": 0.9,
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
                    delay_between_requests: float = 0.1, resume: bool = False):
        """
        Process a JSONL file by making API calls for each prompt.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file (defaults to input_file if not specified)
            delay_between_requests: Delay between API calls in seconds
            resume: Whether to resume from where we left off (skip entries with non-empty responses)
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

        # Process entries
        processed_count = 0
        skipped_count = 0

        for i, entry in enumerate(entries):
            if "prompt" not in entry:
                logger.warning(f"Entry {i} missing 'prompt' field, skipping")
                continue

            # Skip if resuming and response already exists
            if resume and entry.get("response", "").strip():
                skipped_count += 1
                continue

            prompt = entry["prompt"]
            logger.info(f"Processing entry {i+1}/{len(entries)}: ID {entry.get('id', 'N/A')}")

            # Make API call
            response = self.make_api_call(prompt)

            if response:
                entry["response"] = response
                processed_count += 1
                logger.info(f"✓ Generated response ({len(response)} chars)")
            else:
                logger.warning(f"✗ Failed to generate response for entry {i}")
                entry["response"] = ""

            # Add delay between requests to avoid overwhelming the server
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            logger.info(f"✓ Saved results to {output_file}")
            logger.info(f"Summary: Processed {processed_count} entries, Skipped {skipped_count} entries")

        except Exception as e:
            logger.error(f"Failed to save output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL file with vLLM API calls")
    parser.add_argument("--input_file", help="Input JSONL file path", default="leaderboard/lb_prompts.json")
    parser.add_argument("--output", "-o", help="Output JSONL file path (defaults to input file)", default="leaderboard/processed_prompts.json")
    parser.add_argument("--endpoint", "-e", default="http://2080.local:8002/v1/chat/completions",
                       help="vLLM endpoint URL (e.g., http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--model", "-m", help="Model name (optional)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--delay", type=float, default=0,
                       help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing (skip entries with existing responses)")

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
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
