import random


def insert_punctuation(text, frequency=0.1):
        # Define a list of punctuation marks to insert
        punctuation_marks = [',', '.', '!', '?', ';']
        
        # Split the text into words
        words = text.split()
        
        # Calculate the number of punctuations to insert
        num_insertions = max(1, int(len(words) * frequency))
        
        for _ in range(num_insertions):
            # Randomly choose a punctuation mark
            punct = random.choice(punctuation_marks)
            
            # Randomly choose a position to insert the punctuation
            pos = random.randint(0, len(words) - 1)
            
            # Insert the punctuation after the chosen word
            words[pos] += punct
        
        # Join the words back into a single string
        return ' '.join(words)


def augment(texts):
    augmented_texts = []
    for text in texts:
        augmented = insert_punctuation(text)
        augmented_texts.append(augmented)
    return augmented_texts
