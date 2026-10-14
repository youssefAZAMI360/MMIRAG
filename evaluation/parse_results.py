import re

def parse_answer(response_text: str) -> str:
    """Extract the answer letter (A-E) from model output."""
    match = re.search(r"(?:^|\n)\s*([A-E])\b", response_text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-E])\b", response_text)
    return match.group(1) if match else "PARSE_ERROR"
