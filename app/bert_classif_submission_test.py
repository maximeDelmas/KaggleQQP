import os
import pandas as pd
import torch


from transformers import BertTokenizer, logging
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torch.nn import BCELoss

from utils import set_seed, init_model
from bert_cls_classif import BERTSentencesClassificationDataset, BERTSentencesClassification, train_loop_bert_sentences_classification

set_seed(1024)

# Suppress transformers warnings on tokenizer:
logging.set_verbosity_error()

def predictions(model, loader, device, out_dir):
    """Compute probability of duplication for question pairs using a BERTSentencesClassification model.

    Args:
        model (BERTSentencesClassification): the trained model
        test (torch.utils.data.dataloader.DataLoader): the test / validation dataloader
        device (torch.device): the device used by torch cpu or gpu.
        out_dir (str): the output directory
    """

    # Put model in test mode
    model.eval()

    # Init final list
    pred_list = []

    # Iterate over validation batches
    for step, batch in enumerate(loader):

        print(f"step: {step+1} / {len(loader)}")

        # Get batch data
        input_ids = batch[0]['input_ids'].to(device)
        token_type_ids = batch[0]["token_type_ids"].to(device)
        attention_mask = batch[0]["attention_mask"].to(device)

        # Apply model
        out = model(input_ids, token_type_ids, attention_mask)
        out = out.squeeze()

        pred_list += out.tolist()

    df_pred = pd.DataFrame({'test_id': range(len(pred_list)), 'p': pred_list})
    df_pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    DEVICE = torch.device("cpu")

print(DEVICE)

# Params
OUT_DIR = os.path.join("data/submission")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL_PARAMS = dict({'freeze_embedding': True, 'freeze_encoder_layer': 8, 'freeze_cls_pooler': False})
BATCH_SIZE = 128

TEST = pd.read_csv("data/submission/test.csv", index_col=0, low_memory=False)

# On rajoute la colonne is_duplicate pour que le format des données soit pris en charge par le dataset loader
TEST["is_duplicate"] = 0
DATATEST = BERTSentencesClassificationDataset(data=TEST, tokenizer=TOKENIZER, max_length=64)
TEST_LOADER = DataLoader(DATATEST, BATCH_SIZE, shuffle=False)

print("Compute predictions for best-model on validation set")
MODEL = BERTSentencesClassification(**MODEL_PARAMS)
MODEL.load_state_dict(torch.load("data/BERTSentenseClassification/final_train/best-model.pt"))
MODEL.to(DEVICE)
predictions(MODEL, TEST_LOADER, DEVICE, OUT_DIR)