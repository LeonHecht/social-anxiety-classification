import random


def rnd_del(text, deletion_probability=0.2):
        words = text.split()
        if len(words) == 1:  # If the text has only one word, return it as is to avoid empty strings
            return text
        
        # Randomly delete words with the given probability
        filtered_words = [word for word in words if random.random() > deletion_probability]
        
        # If all words are deleted, return a random word (to avoid returning an empty string)
        if len(filtered_words) == 0:
            return random.choice(words)
        
        return ' '.join(filtered_words)


def augment(texts):
    """
    Returns
    -------
    list
        List of texts (str) for the given epoch to train
    """
    return [rnd_del(text) for text in texts]