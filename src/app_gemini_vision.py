import os
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import json
from pydantic import BaseModel
from pathlib import Path
import sys

MODEL = "gemini-2.0-flash-exp"


class ImageDescriptionResponse(BaseModel):
    text: str


def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)


def create_model():
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    return genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config,
    )


def clean_text(text: str) -> str:
    # Remove JSON structure if present
    if text.startswith('{') and text.endswith('}'):
        try:
            text = json.loads(text)['text']
        except:
            pass

    # Split into paragraphs and clean each one
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        # Clean up extra spaces and single newlines within paragraphs
        cleaned = ' '.join(line.strip() for line in paragraph.split('\n'))
        if cleaned:  # Only add non-empty paragraphs
            cleaned_paragraphs.append(cleaned)

    # Join paragraphs with double newlines
    return '\n\n'.join(cleaned_paragraphs)


def get_image_description(image_path: str, model) -> str:
    try:
        image_file = PIL.Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    except PIL.UnidentifiedImageError:
        raise ValueError(f"Invalid image file at: {image_path}")

    prompt = """Extract and return only the original text from the image. 
    Preserve paragraph structure and formatting. 
    Return clean, readable text with proper spacing between paragraphs.
    Do not include any JSON structure or special formatting."""

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [image_file],
            },
        ]
    )

    response = chat_session.send_message(prompt)
    return clean_text(response.text)


def write_to_markdown(text: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    markdown_content = f"## Gemini Vision Results\n\n{text}"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


def main():
    try:
        setup_gemini()
        model = create_model()

        image_path = "./img/mang.jpeg"
        output_path = "./output/from_gemini.md"

        # Get image description
        description = get_image_description(image_path, model)

        # Write to markdown
        write_to_markdown(description, output_path)
        print(f"Successfully wrote results to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
