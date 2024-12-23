from ollama import generate
import base64
import os

MODELS = [
    "llava-phi3",
    "llama3.2-vision",
    "llava-llama3",
    "minicpm-v:8b-2.6-q4_K_M",
    "minicpm-v"
]
IMAGE_FILE = './img/mang.jpeg'
OUTPUT_FILE = './output/from_llm.md'


def read_existing_results():
    if not os.path.exists(OUTPUT_FILE):
        return {}

    results = {}
    current_model = None
    current_text = []

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('## '):
                if current_model:
                    results[current_model] = '\n'.join(current_text).strip()
                current_model = line[3:].replace(' Results\n', '')
                current_text = []
            else:
                current_text.append(line)

    if current_model:
        results[current_model] = '\n'.join(current_text).strip()

    return results


def process_image_with_model(model, image_data):
    try:
        prompt = "Only extract the original writing text from the image out without changing anything."
        response = generate(
            model=model,
            prompt=prompt,
            images=[image_data]
        )
        return response["response"]
    except Exception as e:
        return f"Error processing with {model}: {str(e)}"


def write_results_to_file(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for model, text in results.items():
            f.write(f"## {model} Results\n")
            f.write(f"{text}\n\n")


def main():
    # Read existing results
    existing_results = read_existing_results()

    # Determine which models need to be processed
    models_to_process = [
        model for model in MODELS if model not in existing_results]

    if not models_to_process:
        print("All models have been processed already.")
        return

    # Load and encode image
    with open(IMAGE_FILE, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Process only missing models
    results = existing_results.copy()
    for model in models_to_process:
        print(f"Processing with {model}...")
        results[model] = process_image_with_model(model, image_data)

    # Write combined results to file
    write_results_to_file(results)

    # Print new results to console
    for model in models_to_process:
        print(f"\n=== {model} Results ===")
        print(results[model])


if __name__ == "__main__":
    main()
