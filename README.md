# Large Language and Vision Assistant (LLaVA)
---
*LLaVA* (or Large Language and Vision Assistant), an open-source large multi-modal model, just released version *1.6*. It proposes some improvements over version *1.5*, which was released a few months ago, by:

1. Increasing the input image resolution to *4x* more pixels. This allows it to grasp more visual details. It supports three aspect ratios, up to *672 x 672*, *336 x 1344*, *1344 x 336* resolution.

2. Better visual reasoning and *Optical Character Recognition (OCR)* capability with an improved visual instruction tuning data mixture.

3. Better visual conversation for more scenarios, covering different applications. Better world knowledge and logical reasoning.

4. Efficient deployment and inference with [*SGLang*](https://github.com/sgl-project/sglang).

*LLaVA 1.6* was released in January this year (2024), just 3–4 months after version 1.5. This rapid succession raises the question: How much improvement could be made in such a short time?

Both the initial and new versions are available on [Ollama](https://ollama.com), so I thought I’d give them a try on some of my images to see how they get on.

#### But first, what is the LLaVA Model?

*LLaVA*, or the **L**arge **L**anguage **a**nd **V**ision **A**ssistant, is an advanced *LLM* with enhanced capabilities for recognizing and answering questions about images. Developed by Microsoft, it showcases impressive multimodal abilities, sometimes mimicking the capabilities of advanced models like *GPT-4*.

Designed for general-purpose usage, *LLaVA* can understand and generate responses based on both text and visual inputs. This makes it particularly suitable for applications requiring a nuanced understanding of visual content, such as answering questions about images or interpreting visual data in conversational contexts.

### LLaVA - Overview
![LLaVA](https://raw.githubusercontent.com/sulaiman-shamasna/Large-Language-and-Vision-Assistant-LLaVA/main/images/LLaVA.png)

A significant upgrade in *LLaVA-1.6* is the introduction of the Dynamic High Resolution feature. By increasing the input image resolution to four times more pixels, as mentioned above, the model now supports images up to *672 x 672*, *336 x 1344*, and *1344 x 336* resolutions across three aspect ratios. This enhancement enables the model to capture more visual details, crucial for tasks requiring fine-grained visual understanding.

The implementation of the such advanced technique shown [here](https://llava-vl.github.io/blog/2024-01-30-llava-next/) and [there](https://llava-vl.github.io), allows the model to process various high-resolution images efficiently, using a grid configuration to balance performance and operational costs effectively. This approach significantly reduces the model’s tendency to hallucinate or misinterpret visual content in low-resolution images, thereby improving accuracy and reliability.

*LlaVA-1.6* benefits from an enriched data mixture aimed at improving visual instruction following and conversation capabilities. The model leverages high-quality User Instruct Data, emphasizing diversity in task instructions and the quality of responses. This data mixture includes existing *GPT-V* data sources like *LAION-GPT-V* and *ShareGPT-4V*, alongside a newly curated *15K* visual instruction tuning dataset derived from real-world user requests from the *LLaVA* demo. This dataset is meticulously filtered to address privacy concerns and potential harm, ensuring the model’s responses are both relevant and safe.

### Testing the Models

To compare *Llava 1.6* with version *1.5*, we will have both models interpret the same set of images and analyze the differences in their responses. Here’s the plan:

- **Download [Ollama](https://ollama.com) locally**
- **Set up work environment**

    - In this project, I used **Python 3.10**.
    - Create and activate a virtual environment:
        ```bash
        python -m venv env
        ```
    - And activate it, 
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

- **Install dependencies**
    ```bash
    pip install ollama
    ```

- **Pull LLaVA models using ollama**
    - LLaVA 1.6:
    ```bash
    ollama pull llava
    ```

    - LLaVA 1.5:
    ```bash
    ollama pull llava:7b-v1.5-q4_0
    ```

- **Implementation**
```python
# llava.py
import ollama
from pathlib import Path
from base64 import b64encode

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
    prompt = "May you please give a detailed description of the content of the image."
    
    for model in models:
        print(f"Model: {model}")
        print("=" * 30)
        response = generate_image_description(model, prompt, image_base64)
        print(response, flush=True)
        print("\n")

if __name__ == "__main__":
    main()
```

- **Results**
If you use the Falcon image generated by an *LLM*:

![Falcon](https://raw.githubusercontent.com/sulaiman-shamasna/Large-Language-and-Vision-Assistant-LLaVA/main/images/falcon.png)

The answer to the prompt (**May you please give a detailed description of the content of the image.**) would be:

```bash
Model: llava:7b-v1.5-q4_0
==============================
The image features a large, intricately detailed drawing of an eagle with a yellow beak and eyes. It appears to be the main focus of the scene. Surrounding the eagle are various other elements such as several flames, likely representing its feathers or wings in motion.

In addition to these elements, there is a person visible on the left side of the image who seems to be observing the drawing. It could also be interpreted that their face might be part of the drawing itself.


Model: llava
==============================
This is a digital illustration of a falcon in mid-flight, with its wings fully extended. The falcon has golden-yellow eyes and is depicted in detail, showcasing its textured feathers and sharp beak. Its body is adorned with a fiery pattern that resembles flames, predominantly in shades of yellow, orange, and red, which seem to emanate from the bird chest area. The background is neutral, allowing the bird to stand out as the central subject of the artwork. The falcon is looking slightly to its left. There are no texts or other objects present in the image.
```


