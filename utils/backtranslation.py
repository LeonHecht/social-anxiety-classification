from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
from tqdm import tqdm


def backtranslate(texts, keywords, source_lang="en", intermediate_lang="es", batch_size=10):
    # Load the tokenizer and model for translation to intermediate language
    tokenizer_to_intermediate = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}')
    model_to_intermediate = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}')

    # Load the tokenizer and model for translation back to source language
    tokenizer_to_source = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}')
    model_to_source = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}')

    augmented_texts = []

    for i, text in tqdm(enumerate(texts), desc="Backtranslating", total=len(texts)):
        print("Backtranslating text", i)

        # Translate to intermediate language
        encoded_intermediate = tokenizer_to_intermediate([text], return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            translated_intermediate = model_to_intermediate.generate(**encoded_intermediate)

        intermediate_text = tokenizer_to_intermediate.decode(translated_intermediate[0], skip_special_tokens=True)

        # Translate back to source language
        encoded_source = tokenizer_to_source([intermediate_text], return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            translated_source = model_to_source.generate(**encoded_source)

        backtranslated_text = tokenizer_to_source.decode(translated_source[0], skip_special_tokens=True)

        backtranslated_text += f" (Keywords: {keywords[i]})"

        augmented_texts.append(backtranslated_text)

    return augmented_texts


def backtranslate_t5(texts, keywords, source_lang="en", intermediate_lang="spanish"):
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    # Load the tokenizer and model for translation to intermediate language
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    backtranslated_texts = []

    for i in range(len(texts)):
        print(f"Backtranslating text {i+1} of {len(texts)} texts.")

        # Translate text to intermediate language
        input_text = f"translate english to {intermediate_lang}: {texts[i]}"
        print(f"Text 1: {input_text}")
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        intermediate_outputs = model.generate(input_ids)
        intermediate_text = tokenizer.decode(intermediate_outputs[0], skip_special_tokens=True)

        print(f"Text 2: {intermediate_text}")

        # Translate back to source language
        back_input_text = f"Translate {intermediate_lang} to English: {intermediate_text}"
        back_input_ids = tokenizer.encode(back_input_text, return_tensors="pt", truncation=True, max_length=512)
        back_outputs = model.generate(back_input_ids)
        backtranslated_text = tokenizer.decode(back_outputs[0], skip_special_tokens=True)

        backtranslated_text += f" (Keywords: {keywords[i]})"

        print(f"Text 3: {backtranslated_text}\n\n")

        backtranslated_texts.append(backtranslated_text)

    return backtranslated_texts