
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
# import library for timestamp
from datetime import datetime
from transformers import EarlyStoppingCallback
import torch


# from transformers import TrainerCallback

# class ParaphraseCallback(TrainerCallback):
#     def __init__(self, paraphrase_function, tokenizer):
#         self.paraphrase_function = paraphrase_function
#         self.tokenizer = tokenizer

#     def on_epoch_end(self, args, state, control, **kwargs):
#         print("Paraphrasing training texts after epoch", state.epoch)
#         trainer.train_dataset = paraphrase_dataset(trainer.train_dataset, self.paraphrase_function, self.tokenizer)


# def paraphrase_dataset(dataset, paraphrase_function, tokenizer):
#     texts = [example["text"] for example in dataset]
#     paraphrased_texts = paraphrase_function(texts)
#     new_encodings = tokenizer(paraphrased_texts, truncation=True, padding=True, max_length=128)
#     new_dataset = Dataset.from_dict({"input_ids": new_encodings["input_ids"], "attention_mask": new_encodings["attention_mask"], "labels": dataset["labels"]})
#     return new_dataset


class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            # First, try to access attribute from the DataParallel itself
            return super().__getattr__(name)
        except AttributeError:
            # If failed, try to access it from the wrapped model
            return getattr(self.module, name)


def create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    # Tokenize and encode the text data
    def tokenize_data(texts, labels):
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels}

    train_encodings = tokenize_data(train_texts.to_list(), y_train.to_list())
    val_encodings = tokenize_data(val_texts.to_list(), y_val.to_list())
    test_encodings = tokenize_data(test_texts.to_list(), y_test.to_list())

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)
    
    print("Sample train input_ids:", train_dataset['input_ids'][0])

    return train_dataset, val_dataset, test_dataset        


def load_model(model_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
    # Wrap the model with DataParallel to use multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = CustomDataParallel(model)
    # model.cuda()  # Ensure the model is on the correct device
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
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-6,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False,  # Keep all columns
    )
    return training_args


def get_trainer(model, training_args, train_dataset, val_dataset):
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=4), ParaphraseCallback(paraphrase_function, tokenizer)]
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
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
    return test_pred_labels


def run_optimization(model_checkpoint, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    import optuna
    # num_train_epochs = 3

    def objective(trial):
        # Define the hyperparameters to be optimized
        learning_rate = trial.suggest_float("learning_rate", 5e-7, 5e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 5, 50)

        # Update the training arguments with the suggested hyperparameters
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        trainer = get_trainer(model, training_args, train_dataset, val_dataset)

        # Train the model and get the evaluation results
        trainer.train()
        eval_result = trainer.evaluate()

        # Return the metric to be maximized/minimized
        return eval_result["eval_f1"]

    model = load_model(model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)

    # Run the optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the model with the best hyperparameters
    best_training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=trial.params['num_train_epochs'],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=trial.params["learning_rate"],
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = get_trainer(model, best_training_args, train_dataset, val_dataset)

    # Train the model
    trainer.train()

    # Predict the test dataset
    predictions = predict(trainer, test_dataset)

    # Generate and print the classification report
    test_pred_labels = get_labels(predictions)
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))

    import os
    # print if path "../data/trial_summaries" exists
    print("Trial summary path exists: ", os.path.exists("data/trial_summaries"))

    # Save the trial summary to a CSV file
    sum_df = study.trials_dataframe()
    sum_df.to_csv(f'data/trial_summaries/summary_{model_checkpoint}_{datetime.now()}.csv', index=False)
    print("Trial summary:\n", sum_df)


def run_lora(model_checkpoint, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)
    model = load_model(model_checkpoint)
    print(model)

    # LoRA config
    # target_modules=["q_proj", "v_proj"],
        # target_modules = [
        #     layer_name
        #     for i in range(48)  # Adjust based on the number of layers in your model
        #     for layer_name in [
        #         f"deberta.encoder.layer.{i}.attention.self.query_proj",
        #         f"deberta.encoder.layer.{i}.attention.self.key_proj",
        #         f"deberta.encoder.layer.{i}.attention.self.value_proj",
        #         f"deberta.encoder.layer.{i}.attention.output.dense"
        #     ]
        # ],
    lora_config = LoraConfig(
        target_modules = [
            f"distilbert.transformer.layer.{i}.attention.q_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.k_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.v_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.attention.out_lin" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.ffn.lin1" for i in range(6)] + [
            f"distilbert.transformer.layer.{i}.ffn.lin2" for i in range(6)
        ],
        r=64,
        task_type="SEQ_CLS",
        lora_alpha=128,
        lora_dropout=0.05,       # 0.05
        use_rslora=True
    )

    # load LoRA model
    lora_model = get_peft_model(model, lora_config)

    training_args = training_arguments()
    trainer = get_trainer(lora_model, training_args, train_dataset, val_dataset)
    
    # Train the model
    print("Training the model...")
    print("Verifying train dataset structure...")
    print(train_dataset[0])  # This should print the first element to check structure
    trainer.train()

    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    # Generate and print the classification report
    print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return test_pred_labels