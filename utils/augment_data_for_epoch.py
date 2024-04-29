from utils import punct_insertion
from utils import random_insertion
from utils import random_deletion
from utils import random_swap


def augment(epoch, texts):
    """
    Modifies the text based on the augmentation strategy determined by the epoch number.

    Parameters
    ----------
    epoch : int
        Current training epoch number.
    texts : list of str
        List of text strings to augment.

    Returns
    -------
    list of str
        List of augmented texts for the given epoch to train.

    Raises
    ------
    ValueError
        If the epoch is 5 or less, as augmentation should start after epoch 5.
    """
    if epoch <= 5:
        raise ValueError("Epoch should be greater than 5 for augmentation")

    # Calculate the cycle index for augmentation
    # Epochs 6-9 correspond to indices 0-3
    cycle_index = (epoch - 6) % 4

    if cycle_index == 0:
        texts = punct_insertion.augment(texts)
    elif cycle_index == 1:
        texts = random_insertion.augment(texts)
    elif cycle_index == 2:
        texts = random_deletion.augment(texts)
    elif cycle_index == 3:
        texts = random_swap.augment(texts)

    return texts
