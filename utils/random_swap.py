import random


def rnd_swap(text, n_swaps=1):
        words = text.split()
        length = len(words)
        if length < 2:
            return text  # Return the text as is if it's too short to swap
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(length), 2)  # Get two distinct indices
            words[idx1], words[idx2] = words[idx2], words[idx1]  # Swap the words at these indices
        
        return ' '.join(words)


def augment(texts):
    """
    Returns
    -------
    list
        List of texts (str) for the given epoch to train
    """
    return [rnd_swap(text) for text in texts]