
import numpy as np
import random
import torch
from transformers.file_utils import is_torch_available

def fix_dataset(dataset):
    
    # Check is all questions in 'question1' and 'question2' are str
    filter = np.array([not isinstance(s1, str) for s1 in dataset['question1'].tolist()]) | np.array([not isinstance(s2, str) for s2 in dataset['question2'].tolist()])
    indexes_to_drop = dataset[filter].index
    
    # drop lines that are not
    if not len(indexes_to_drop):
        print("All rows are corrects")
    else:
        print("Removing the following lines: ")
        print(dataset.loc[indexes_to_drop])
        dataset = dataset.drop(indexes_to_drop)
    
    return dataset



def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
