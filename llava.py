import ollama
from pathlib import Path
from base64 import b64encode
import os

def load_image_as_base64(image_path):

    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return b64encode(image_bytes).decode('utf-8')

def generate_image_description(model, prompt, image_base64):

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_base64], 
        stream=True
    )
    
    response = ""
    for chunk in stream:
        response += chunk['response']
    return response

def main():
    models = ["llava:7b-v1.5-q4_0", "llava"]
    img_name = "falcon.png"
    image_path = os.path.join("images", img_name)
    image_path = Path(image_path)   
    image_base64 = load_image_as_base64(image_path)  
    prompt = "Describe in detail what this image is"
    
    for model in models:
        print(f"Model: {model}")
        print("=" * 30)
        response = generate_image_description(model, prompt, image_base64)
        print(response, flush=True)
        print("\n")

if __name__ == "__main__":
    main()
