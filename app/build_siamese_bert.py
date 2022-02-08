import os
import pandas as pd
import torch


from transformers import BertTokenizer
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from utils import set_seed, init_model
from siamese_bert import ContrastiveLoss, SiameseNetWorkSentenceDataset, train_loop_siamese, SiameseBERTNet, pwdist_siamese_bert

set_seed(1024)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    DEVICE = torch.device("cpu")

print(DEVICE)

# Params
OUT_DIR = os.path.join("data/SiameseBERT/batch128_m10_training")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
BATCH_SIZE = 128
NEPOCHS = 10
MODEL_PARAMS = dict({'freeze_embedding': True, 'freeze_encoder_layer': 8, 'freeze_cls_pooler': True})
LOSS = ContrastiveLoss(m=10)


# Load data
CV = pd.read_csv("data/SiameseBERT/CV/cv_data_100000.csv", index_col=False)
DATASET = SiameseNetWorkSentenceDataset(data=CV, tokenizer=TOKENIZER, max_length=64)

# CV split

KFOLD = KFold(n_splits=5, shuffle=True, random_state=1)
TRAIN_IDS, VAL_IDS = next(iter(KFOLD.split(DATASET)))
TRAIN_LOADER = DataLoader(DATASET, BATCH_SIZE, sampler=SubsetRandomSampler(TRAIN_IDS))
VAL_LOADER = DataLoader(DATASET, BATCH_SIZE, sampler=SubsetRandomSampler(VAL_IDS))

# Init & load model

MODEL = SiameseBERTNet(**MODEL_PARAMS)
MODEL.to(DEVICE)
OPT, SCHED = init_model(MODEL, TRAIN_LOADER, NEPOCHS)

# Train
LOGS = train_loop_siamese(model=MODEL, dataloader=TRAIN_LOADER, validation=VAL_LOADER, optimizer=OPT, scheduler=SCHED, loss_fn=LOSS, eval_threshold=5, nepochs=NEPOCHS, device=DEVICE, out_dir=OUT_DIR, save=True)

# Save logs
LOGS.to_csv(os.path.join(OUT_DIR, "training_logs.csv"), index=True, header=True)

# Pred on val
MODEL = SiameseBERTNet(**MODEL_PARAMS)
MODEL.load_state_dict(torch.load(os.path.join(OUT_DIR, "best-model.pt")))
MODEL.to(DEVICE)
pwdist_siamese_bert(MODEL, VAL_LOADER, DEVICE, OUT_DIR)
