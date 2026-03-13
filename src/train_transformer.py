# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer


# Load Dataset
df = pd.read_csv("data/raw/train.csv")

texts = df["text"]
labels = df["Sarcasm"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)


# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "text": list(X_train),
    "label": list(y_train)
})

test_dataset = Dataset.from_dict({
    "text": list(X_test),
    "label": list(y_test)
})

# Load Tokenizer
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize Text


def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True
    )


train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Training Configuration

training_args = TrainingArguments(

    output_dir="../models/bert",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    logging_dir="../results"
)


# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train Model
trainer.train()


# Save Model
# trainer.save_model("../models/bert")

# Train Other Transformers
model2_name = "roberta-base"
model3_name = "distilbert-base-uncased"
