import os
import logging
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError, ServerError

load_dotenv()

# Log to file assignment_3.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="assignment_3.log",
    filemode="a",
)

# Exit if missing API key
if not os.getenv("GEMINI_API_KEY"):
    raise KeyError("Gemini API key not found")

# Constants
MAX_WORDS = 200
MODEL = "gemini-2.0-flash-001"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def truncate_text(text: str) -> str:
    """Truncate text to maximum number of words."""
    words = text.split()
    if len(words) > MAX_WORDS:
        logging.info(f"Text length: {len(words)}. Truncating to {MAX_WORDS}...")
        return " ".join(words[:MAX_WORDS]) + "..."

    return text


def summarize_text(text: str) -> Optional[str]:
    """
    Summarize input text, returning a stream so user feels like output is faster. Rate limiting errors are handled in ClientError response code as per docs/HTTP conventions:

    https://github.com/googleapis/python-genai/blob/1520726dc64e9cf757ef07ceb354fda104be1017/google/genai/errors.py
    """
    try:
        chat = client.chats.create(model=MODEL)

        prompt = f"Summarize the following text: {text}"

        summary = ""
        for chunk in chat.send_message_stream(prompt):
            if hasattr(chunk, "text"):
                summary += chunk.text
                print(chunk.text, end="", flush=True)

        return summary

    except ClientError as e:
        logging.error(f"A Client error occurred (400-499 response code): {e}")

    except ServerError as e:
        logging.error(f"A Server error occurred (500-599 response code): {e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None


def main():
    input_text = input(f"Enter text to summarize (max {MAX_WORDS} words): ")
    truncated_text = truncate_text(input_text)
    print("\\nnSummary:\n\n")
    complete_output = summarize_text(truncated_text)
    print(f"\n\nComplete output below: \n\n{complete_output}")


if __name__ == "__main__":
    main()
