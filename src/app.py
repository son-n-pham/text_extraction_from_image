# import sys
# import ollama
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OLLAMA_GPU"] = "1"
# os.environ["OLLAMA_FLASH_ATTENTION"] = "1"


# PROMPT = """
# Extract writing text from the image.
# """

# FILE = 'mang.jpeg'
# OUTPUT_FILE = 'student_mang.md'

# try:
#     response = ollama.chat(
#         model='llama3.2-vision',  # Fixed model name
#         messages=[{
#             'role': 'user',
#             'content': PROMPT,
#             'images': [FILE]
#         }]
#     )

#     print(response)

#     # Access content via message object
#     content = response.message.content

#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         f.write(content)

#     print(f"Response written to {OUTPUT_FILE}")

# except ollama._types.ResponseError as e:
#     print(f"Ollama error: {e}")
#     print("Make sure to pull the model first using: ollama pull llama2-vision")
#     sys.exit(1)
# except Exception as e:
#     print(f"Error: {e}")
#     sys.exit(1)

import ollama
import os
import sys


def setup_gpu_environment():
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["OLLAMA_GPU"] = "1"
            os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
            print("GPU enabled for Ollama")
            return True
    except ImportError:
        pass
    print("Running in CPU mode")
    return False


def process_image(image_path, prompt):
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        return response.message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    PROMPT = "Extract writing text from the image."
    FILE = 'mang.jpeg'
    OUTPUT_FILE = 'output.md'

    # Setup environment
    setup_gpu_environment()

    # Process image
    content = process_image(FILE, PROMPT)
    if content:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Response written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
