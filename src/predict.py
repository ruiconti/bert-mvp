import os
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

HF_PRETRAINED = "rohanrajpal/bert-base-codemixed-uncased-sentiment"
FINETUNED = 'model/bert-npl'
CLASSES = ('baixa', 'media',  'alta')

def load_model(path: str):
   assert os.path.exists(path)
   model = AutoModelForSequenceClassification.from_pretrained(path)
   return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained(HF_PRETRAINED)

tokenizer = load_tokenizer()
model = load_model(FINETUNED)

def evaluate(text, confidence_score=False):
  encoded_predict_input = tokenizer(
      text,
      truncation=True,
      padding=True,
      return_tensors="pt"
    )
  result = torch.softmax(model(**encoded_predict_input)[0], dim=1).tolist()[0]
  max_result = max(result)
  max_result_idx = result.index(max_result)
  max_result_class = CLASSES[max_result_idx]

  return max_result_idx if confidence_score is False else (
      max_result_idx, dict(zip(CLASSES, result))
    )
