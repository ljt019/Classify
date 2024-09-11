# Sketch Classifier CLI

This is a command-line tool for classifying sketch images using a pre-trained Vision Transformer (ViT) model.

## Features

- Uses a ViT model fine-tuned for sketch classification
- Provides top 5 predictions with confidence scores
- Colorized output for better readability

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/sketch-classifier-cli.git
   cd sketch-classifier-cli
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Install the tool:
   ```
   pip install .
   ```

## Usage

After installation, you can use the tool from anywhere in your system by running:

```
classify path/to/your/sketch.jpg
```

This will classify the sketch image and output the results, including:

- The top prediction with its class ID
- A table of the top 5 predictions with their confidence scores

## Example

```
$ classify dog_sketch.jpg

==================================================
🖼️  Image: dog_sketch.jpg
🏷️  Prediction: dog
🔢  Id: 66
==================================================

Top 5 Predictions:
Rank  Label               ID    Confidence
------------------------------------------
1.    dog                 66    85.32%
2.    cat                 48    10.45%
3.    horse               106   2.67%
4.    cow                 59    1.12%
5.    sheep               185   0.44%

==================================================
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pillow
- termcolor
