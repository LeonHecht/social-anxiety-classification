
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerControl, TrainerState
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
# import library for timestamp
from datetime import datetime
from transformers import EarlyStoppingCallback
import torch
from transformers import TrainerCallback

from sklearn.utils.class_weight import compute_class_weight


class WeightedAutoModel(AutoModelForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def compute_loss(self, model_output, labels):
        # model_output: tuple of (logits, ...)
        logits = model_output[0]
        # Assuming using CrossEntropyLoss, adjust accordingly if using a different loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


class LossDifferenceCallback(TrainerCallback):
    def __init__(self, loss_diff_threshold):
        # Threshold for difference in loss
        self.loss_diff_threshold = loss_diff_threshold
        self.training_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Store training loss from each logging step
        if logs is not None:
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        # Calculate the average training loss
        average_training_loss = sum(self.training_losses) / len(self.training_losses) if self.training_losses else float('inf')
        # Get the validation loss from the evaluation metrics
        validation_loss = metrics.get("eval_loss", float("inf"))
        # print the average training loss and validation loss with two decimal places
        print(f"\n\nAverage training loss: {average_training_loss:.2f}, Validation loss: {validation_loss:.2f}\n\n")

        # Calculate the difference and decide if training should stop
        loss_diff = validation_loss - average_training_loss
        if loss_diff > self.loss_diff_threshold:
            print(f"Stopping training due to loss difference: {loss_diff}")
            control.should_training_stop = True

        # Reset training losses after evaluation
        self.training_losses = []


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


def load_model(model_checkpoint, num_labels, classes, y_train):
    # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    weighted_model = WeightedAutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
    weighted_model.class_weights = class_weights_tensor
    # weighted_model.load_state_dict(model.state_dict())  # Copy the weights from the original model
    # Wrap the model with DataParallel to use multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = CustomDataParallel(model)
    # model.cuda()  # Ensure the model is on the correct device
    return weighted_model


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
    weight_decay=0.02,
    learning_rate=5e-6,
    logging_dir='./logs',
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
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
    callbacks=[LossDifferenceCallback(loss_diff_threshold=0.2)]
    )
    return trainer


def predict(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    return predictions


def get_labels(predictions):
    test_pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("Predicted Labels", test_pred_labels)
    return test_pred_labels

def run(model_checkpoint, num_labels, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test)
    classes = np.unique(y_train)
    print("Type of classes:", type(classes))
    print("Classes:", classes)
    model = load_model(model_checkpoint, num_labels, classes, y_train)
    training_args = training_arguments()
    trainer = get_trainer(model, training_args, train_dataset, val_dataset)
    # Train the model
    trainer.train()
    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    # Generate and print the classification report
    if num_labels == 2:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return test_pred_labels


def run_optimization(model_checkpoint, train_texts, val_texts, test_texts, y_train, y_val, y_test):
    import optuna
    # num_train_epochs = 3

    def objective(trial):
        # Define the hyperparameters to be optimized
        learning_rate = trial.suggest_float("learning_rate", 5e-7, 5e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 5, 50, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 128, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.1, 0.3)

        # Update the training arguments with the suggested hyperparameters
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=weight_decay,
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
        per_device_train_batch_size=trial.params['batch_size'],
        per_device_eval_batch_size=trial.params['batch_size'],
        warmup_steps=500,
        weight_decay=trial.params['weight_decay'],
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