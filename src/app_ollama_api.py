import requests
import json
import base64
from typing import Dict, Generator, List


class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434", debug: bool = False):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        self.debug = debug

    def generate(self, prompt: str, model: str = "phi3.5:latest", stream: bool = False, stop_sequence: str = "<|end|>") -> Dict:
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "stop": stop_sequence
        }
        try:
            response = requests.post(
                url, headers=self.headers, json=data, stream=stream)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def generate_stream(self, prompt: str, model: str = "phi3.5:latest", stop_sequence: str = "<|end|>") -> Generator[str, None, None]:
        response = self.generate(
            prompt, model, stream=True, stop_sequence=stop_sequence)

        try:
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    if self.debug:
                        print(f"Debug: {json_response}")
                    chunk = json_response.get('response', '')
                    yield chunk
                    if stop_sequence in chunk:
                        break
        except Exception as e:
            raise Exception(f"Streaming error: {str(e)}")

    def generate_with_image(self, prompt: str, image_data: str, model: str = "llama3.2-vision") -> Dict:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ]
        }
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return encoded_string


def main():
    input_prompt = "Give me a joke"
    image_prompt = "what is in this image?"
    # Replace with the actual path to your image
    image_path = "mang.jpeg"

    client = OllamaAPI(debug=False)
    try:
        print(f"Input: {input_prompt}")
        print("Response: ", end='', flush=True)
        for chunk in client.generate_stream(input_prompt):
            print(chunk.replace("<|end|>", ""), end='', flush=True)
        print()  # New line after response

        image_data = client.encode_image_to_base64(image_path)
        print(f"Image Input: {image_prompt}")
        response = client.generate_with_image(image_prompt, image_data)
        print(f"Image Response: {response}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
