from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def paraphrase(texts, keywords):

    # Define a function for paraphrasing
    def paraphrase_text(text, keyword, max_length=60):
        # Encode the input text
        print("Type of text:", type(text))
        print("text:", text)

        input_ids = tokenizer.encode(f"paraphrase (your response should include the outdoor activity keyword '{keyword}'): " + text, return_tensors="pt", truncation=True, max_length=max_length)

        # Generate paraphrases
        paraphrases = model.generate(input_ids, do_sample=True, temperature=1.5, num_beams=100, num_return_sequences=5, max_length=max_length)

        # Decode the generated paraphrases
        paraphrased_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in paraphrases]

        return paraphrased_texts
    
    # Load the model and tokenizer
    model_name = "tuner007/pegasus_paraphrase"
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    
    paraphrased_texts_final = []
    for i in range(len(texts)):
        print(f"Paraphrasing text {i}/{len(texts)}")
        paraphrased_texts = paraphrase_text(texts[i], keywords[i])
        print(f"paraphrased_texts {i}:", paraphrased_texts)
        paraphrased_texts_final.append(paraphrased_texts)
    print("paraphrased_texts_final:\n", paraphrased_texts_final)
    return paraphrased_texts_final