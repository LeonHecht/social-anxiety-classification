from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def paraphrase(texts):
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

    def paraphrase_text(
        question,
        num_beams=10,
        num_beam_groups=10,
        num_return_sequences=10,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    ):
        input_ids = tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        outputs = model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res
    
    paraphrased_texts_final = []
    for i in range(len(texts)):
        print(f"Paraphrasing text {i}/{len(texts)}")
        print("Current text:", texts[i])
        paraphrased_texts = paraphrase_text(texts[i])
        print(f"paraphrased_texts {i}:", paraphrased_texts)
        paraphrased_texts_final.append(paraphrased_texts)
    print("paraphrased_texts_final:\n", paraphrased_texts_final)
    return paraphrased_texts_final