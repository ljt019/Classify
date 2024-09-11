import argparse
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AutoConfig
from termcolor import colored
import sys
import io

# Set console output encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

def load_model_and_config(model_name):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Load the configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Initialize the processor with the config
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Use the model's built-in id2label mapping
    id2label = model.config.id2label
    
    # Ensure all keys are integers
    id2label = {int(k): v for k, v in id2label.items()}
    
    return model, processor, id2label

def preprocess_image(image_path, processor):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def predict(image_path, model, processor):
    inputs = preprocess_image(image_path, processor)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = logits.argmax(-1).item()
    return predicted_class, probabilities

def main():
    parser = argparse.ArgumentParser(description="Classify an image using a pre-trained model.")
    parser.add_argument("image_path", help="Path to the image file to classify.")
    args = parser.parse_args()

    model_name = "valentingerard100/vit-base-patch16-224-in21k-finetuned-ViT-sketches"
    model, processor, id2label = load_model_and_config(model_name)

    predicted_class, probabilities = predict(args.image_path, model, processor)

    predicted_label = id2label[predicted_class]

    # Print the main prediction
    print("\n" + "="*50)
    print(f"🖼️  Image: {args.image_path}")
    print(f"🏷️  Prediction: {colored(predicted_label, 'cyan', attrs=['bold'])}")
    print(f"🔢  Id: {predicted_class}")
    print("="*50 + "\n")

    # Print top 5 predictions in a tabular format
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(colored("Top 5 Predictions:", attrs=['underline']))
    print(f"{'Rank':<6}{'Label':<20}{'ID':<6}{'Confidence':<10}")
    print("-"*42)
    for i in range(top5_prob.size(1)):
        class_id = top5_catid[0, i].item()
        label = id2label[class_id]
        prob_percent = top5_prob[0, i].item() * 100
        rank = f"{i+1}."
        confidence = f"{prob_percent:.2f}%"
        
        # Color-code the confidence levels
        if i == 0:
            confidence = colored(confidence, 'green', attrs=['bold'])
        elif prob_percent > 15:
            confidence = colored(confidence, 'yellow')
        
        print(f"{rank:<6}{label:<20}{class_id:<6}{confidence:<10}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()