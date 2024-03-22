from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
from tqdm import tqdm


def backtranslate(texts, source_lang="en", intermediate_lang="es", batch_size=10):
    # Load the tokenizer and model for translation to intermediate language
    tokenizer_to_intermediate = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}')
    model_to_intermediate = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}')

    # Load the tokenizer and model for translation back to source language
    tokenizer_to_source = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}')
    model_to_source = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}')

    augmented_texts = []

    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches), desc="Backtranslating", total=num_batches):
        if i == num_batches - 1:
            batch_texts = texts[i * batch_size:]
        else:
            batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        print("Backtranslating text", i * batch_size, "to", (i + 1) * batch_size)
        print("Len batch texts:", len(batch_texts))

        # Translate to intermediate language
        encoded_intermediate = tokenizer_to_intermediate(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated_intermediate = model_to_intermediate.generate(**encoded_intermediate)

        intermediate_texts = [tokenizer_to_intermediate.decode(t, skip_special_tokens=True) for t in translated_intermediate]

        # Translate back to source language
        encoded_source = tokenizer_to_source(intermediate_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated_source = model_to_source.generate(**encoded_source)

        backtranslated_texts = [tokenizer_to_source.decode(t, skip_special_tokens=True) for t in translated_source]

        augmented_texts.extend(backtranslated_texts)

    return augmented_texts