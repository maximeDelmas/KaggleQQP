import random
import numpy as np
import torch
from transformers.file_utils import is_torch_available
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

def fix_dataset(dataset):
    """Fix dataset issues. Some rows don't have 2 questions for instance.
    """

    # Check is all questions in 'question1' and 'question2' are str
    rows_to_delete = np.array([not isinstance(s1, str) for s1 in dataset['question1'].tolist()]) | np.array([not isinstance(s2, str) for s2 in dataset['question2'].tolist()])
    indexes_to_drop = dataset[rows_to_delete].index
    print(indexes_to_drop)
    print(type(indexes_to_drop))

    # drop lines
    if indexes_to_drop.empty:
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

def check_params(model):
    """Check model parameters setting

    Args:
        model (nn.Module]): A torch model
    """
    trainable_sum, non_trainable_sum = 0, 0
    for name, param in model.named_parameters():
        nb_parameters = param.numel()
        rq_grad = param.requires_grad
        print(f"{name:>60} | {nb_parameters:>9} | {str(rq_grad):>6}")
        if rq_grad:
            trainable_sum += nb_parameters
        else:
            non_trainable_sum += nb_parameters
    print("Total number of trainaible parameters: " + str(trainable_sum))
    print("Total number of non-trainable parameters: " + str(non_trainable_sum))


def init_model(model, dataloader, nepochs):
    """Initialize optimzer and scheduler for training

    Args:
        model (nn.Module): the model to train
        dataloader (torch.utils.data.dataloader.DataLoader): the training dataLoader
        nepochs (int): number of epochs for training

    Returns:
        [transformers.AdamW, torch.optim.lr_scheduler.LambdaLR]: the parametrized optimizer and scheduler
    """

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
        lr=5e-5,    # Default learning rate
        eps=1e-8    # Default epsilon value
        )

    # Get total number of steps
    nbatchs = len(dataloader)
    total_nb_steps = nbatchs * nepochs

    # Create the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=0, # Default value so direct training without warmup
        num_training_steps=total_nb_steps)

    return optimizer, scheduler
