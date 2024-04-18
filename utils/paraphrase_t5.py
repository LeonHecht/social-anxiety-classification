from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def paraphrase(texts):
    def paraphrase_text(text, num_samples=10, max_length=60):
        model_name = "t5-base"  # Example: Change to your model, e.g., "facebook/mbart-large-cc25"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Prepare the text
        input_ids = tokenizer.encode("paraphrase: " + text, return_tensors="pt", truncation=True, max_length=max_length)

        # Generate paraphrases
        paraphrases = model.generate(
            input_ids,
            num_return_sequences=num_samples,
            num_beams=num_samples * 2,
            early_stopping=True,
            max_length=max_length,
        )

        # Decode generated ids to text
        return [tokenizer.decode(g, skip_special_tokens=True) for g in paraphrases]

    paraphrased_texts_final = []
    for i in range(len(texts)):
        print(f"Paraphrasing text {i}/{len(texts)}")
        print("Current text:", texts[i])
        paraphrased_texts = paraphrase_text(texts[i])
        print(f"paraphrased_texts {i}:", paraphrased_texts)
        paraphrased_texts_final.append(paraphrased_texts)
    print("paraphrased_texts_final:\n", paraphrased_texts_final)
    return paraphrased_texts_final

if __name__ == "__main__":
    # Example usage
    text = "The quick brown fox jumps over the lazy dog."
    paraphrased_texts = paraphrase(text)
    for i, paraphrase in enumerate(paraphrased_texts, start=1):
        print(f"Paraphrase {i}: {paraphrase}")
