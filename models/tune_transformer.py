
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report


def create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    # Tokenize and encode the text data
    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels}

    train_encodings = tokenize_data(train_texts.to_list(), y_train.to_list())
    val_encodings = tokenize_data(val_texts.to_list(), y_val.to_list())
    test_encodings = tokenize_data(test_texts.to_list(), y_test.to_list())

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)
    
    return train_dataset, val_dataset, test_dataset


def load_model(model_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def training_arguments():
    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
    )
    return training_args


def get_trainer(model, training_args, train_dataset, val_dataset):
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
    )
    return trainer


def predict(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    return predictions


def get_labels(predictions):
    test_pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("Predicted Labels", test_pred_labels)
    return test_pred_labels

def run(model_checkpoint, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    # 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)
    model = load_model(model_checkpoint)
    training_args = training_arguments()
    trainer = get_trainer(model, training_args, train_dataset, val_dataset)
    # Train the model
    trainer.train()
    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    
    # Generate and print the classification report
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))