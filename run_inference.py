"""A module for the inference of the model."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PyQt6.QtWidgets import (QApplication, QLabel, QLineEdit, QPushButton,
                             QTextEdit, QVBoxLayout, QWidget)
from torch import dtype
from transformers import (AutoTokenizer,
                          T5ForConditionalGeneration)


def get_current_torch_device() -> str:
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(model_path: str | Path, torch_dtype: dtype = torch.float32,
               lora_path: str | Path = "") \
        -> tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """
    Load a model from a checkpoint.

    :param model_path: The path to the checkpoint.
    :param torch_dtype: The precision in which to load the model.
    :param lora: The path to the LoRa checkpoint.
    :return: The model and the tokenizer.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.to(get_current_torch_device())

    return model, tokenizer


def run_inference(model: T5ForConditionalGeneration, tokenizer: AutoTokenizer,
                  input_texts: list[str], max_length: int = 100) -> list[str]:
    """
    Run batched inference.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param input_texts: The input texts.
    :param max_length: The maximum length of the output.
    :return: The generated texts.
    """
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)

    output_sequences = model.generate(
        input_ids=inputs.input_ids,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        max_length=max_length,
        num_beams=3
    ).cpu()

    outputs = np.where(output_sequences != -100, output_sequences, tokenizer.pad_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


class DefinitionGeneratorApp(QWidget):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Definition Generator')
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.wordLabel = QLabel('Word:')
        layout.addWidget(self.wordLabel)

        self.wordInput = QLineEdit()
        layout.addWidget(self.wordInput)

        self.contextLabel = QLabel('Context:')
        layout.addWidget(self.contextLabel)

        self.contextInput = QLineEdit()
        layout.addWidget(self.contextInput)

        self.generateButton = QPushButton('Generate Definition')
        self.generateButton.clicked.connect(self.onGenerateClicked)
        layout.addWidget(self.generateButton)

        self.resultText = QTextEdit()
        self.resultText.setReadOnly(True)
        layout.addWidget(self.resultText)

        self.setLayout(layout)

    def onGenerateClicked(self):
        word = self.wordInput.text()
        context = self.contextInput.text()

        if word and context:
            input_text = f"<LM> Контекст: \"{context}\" Определение слова \"{word}\": "
            generated_definition = run_inference(self.model, self.tokenizer, [input_text])
            self.resultText.setText(generated_definition[0])
        else:
            self.resultText.setText("Please enter both a word and a context.")


def main():
    parser = argparse.ArgumentParser(description="Definition Generator with GUI")
    parser.add_argument("model_path", type=str,
                        help="The path to the model checkpoint.")
    parser.add_argument("-l", "--lora_path", type=str, default="",
                        help="The path to the LoRa checkpoint.")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    model, tokenizer = load_model(args.model_path, lora_path=args.lora_path)
    ex = DefinitionGeneratorApp(model, tokenizer)
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
