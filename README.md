# Vietnamese Hate Speech Detection with PhoBERT

This project fine-tunes the **PhoBERT** base model (`vinai/phobert-base-v2`) to classify Vietnamese text into three categories: **Clean**, **Offensive**, and **Hate**. It utilizes the `vominhmanh/vihsd-explainable` dataset and the Hugging Face `transformers` library for training, evaluation, and inference.

## Features

  * **Custom Preprocessing**: Uses `underthesea` for Vietnamese word segmentation to match PhoBERT's expected input format.
  * **Model Training**: Leverages the Hugging Face `Trainer` API with early evaluation and macro F1-score tracking.
  * **Performance Evaluation**: Generates a detailed classification report and a visual confusion matrix using `scikit-learn` and `seaborn`.
  * **Google Drive Integration**: Automatically saves and loads the fine-tuned model to/from Google Drive.
  * **Interactive Inference**: Includes a ready-to-use function to predict hate speech on raw text.

## Requirements

To run the notebook, you need the following dependencies installed in your Python environment:

```bash
pip install transformers[torch] datasets evaluate accelerate underthesea scikit-learn matplotlib seaborn
```

## Model Performance

Based on the test set evaluation, the model achieved the following overall metrics:

  * **Accuracy**: \~88%
  * **Macro F1-Score**: \~0.66

*Note: The model performs exceptionally well on "Clean" text (F1: 0.94) but shows moderate performance on identifying specific "Offensive" (F1: 0.43) and "Hate" (F1: 0.61) classes due to class imbalances typical in hate speech datasets.*

## Dataset

The model is trained on the [vominhmanh/vihsd-explainable](https://www.google.com/search?q=https://huggingface.co/datasets/vominhmanh/vihsd-explainable) dataset, a corpus designed for Vietnamese hate speech detection.

## How to Use

The provided Jupyter Notebook runs sequentially.

1.  **Train**: Sets up the tokenizers, filters the dataset, and runs the Hugging Face `Trainer`.
2.  **Evaluate**: Tests the dataset and plots a heatmap to analyze model accuracy per class.
3.  **Save**: Stores the trained weights (`/content/drive/MyDrive/phobert_hate_speech_model`).
4.  **Load & Predict**: Connects to the saved model and executes the inference function.

### Running Inference on Your Own Text

After running the notebook and loading the model, you can test it on any Vietnamese text. **To test your own quote, simply modify the `test_text` variable at the bottom of the notebook.**

```python
# --- Interactive Test ---
# Change the quote below to test your own text!
test_text = "Bạn ngu vcl" 
label, confidence = predict_hate_speech(test_text)

print(f"Text: {test_text}")
print(f"Result: {label} ({confidence*100:.2f}%)")
```

*Current output for the default text*: `Result: Offensive (94.51%)`
