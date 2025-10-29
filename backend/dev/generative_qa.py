""" 
Creating the architecture to get a response from an LLM

Running this on Colab only -- way too slow on CPU

"""

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, pipeline


def preprocess_function(examples):
    inputs = [
        f"context: {c} question: {q}"
        for c, q in zip(examples["context"], examples["question"])
    ]
    targets = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
    return {"input_text": inputs, "target_text": targets}


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

dataset = load_dataset("squad")
processed_dataset = dataset.map(preprocess_function, batched=True)
# df = pd.DataFrame(processed_dataset["train"])
# df.to_csv("dev/data/squad_train.csv", index=False)
# data = pd.read_csv("dev/data/squad_train.csv")
# print(data)
model_name = "google/flan-t5-small"  # or "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
max_input_length = 512
max_target_length = 64
tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="results/development/models/flan-t5-squad",
    eval_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=50
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
    )

trainer.train()
trainer.save_model("results/development/models/flan-t5-squad-final")
tokenizer.save_pretrained("results/development/models/flan-t5-squad-final")


qa_pipeline = pipeline("text2text-generation", model="results/development/models/flan-t5-squad-final", tokenizer=tokenizer)

question = "Who wrote Pride and Prejudice?"
context = "Pride and Prejudice is a novel written by Jane Austen in 1813."

prompt = f"context: {context} question: {question}"
answer = qa_pipeline(prompt)[0]["generated_text"]

print("Answer:", answer)