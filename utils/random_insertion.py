import random
from nltk.corpus import wordnet


def rnd_insert(text, n=4):
        words = text.split()
        new_words = words.copy()
        for _ in range(n):
            synonyms = []
            word_to_replace = random.choice(words)
            
            # Find synonyms using WordNet
            for syn in wordnet.synsets(word_to_replace):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())  # add the synonyms
            
            # Remove any duplicates or the original word itself
            synonyms = set(synonyms)
            synonyms.discard(word_to_replace)
            
            if synonyms:
                synonym = random.choice(list(synonyms))
                # Randomly choose a position to insert the synonym
                position = random.randint(0, len(new_words))
                new_words.insert(position, synonym)
        
        return ' '.join(new_words)


def augment(texts):
    """
    Returns
    -------
    list
        List of texts (str) for the given epoch to train
    """
    return [rnd_insert(text) for text in texts]