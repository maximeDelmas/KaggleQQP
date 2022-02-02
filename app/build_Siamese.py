import torch
import numpy as np
import pandas as pd
import os

from siameseBERT import *
from utils import *

set_seed(1024)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

device = torch.device("cpu")
print(device)

data_path = "data/quora-question-pairs"
train_file = "train.csv"
test_pos_file = "test.csv"
label_file = "sample_submission.csv"

# Reading
train = pd.read_csv(os.path.join(data_path, train_file), index_col = 0)
test_pos = pd.read_csv(os.path.join(data_path, test_pos_file), index_col = 0)
y_label = pd.read_csv(os.path.join(data_path, label_file), index_col = 0)

# Fix datasets for NaN values in question1 or question2
train = fix_dataset(train)
test_pos = fix_dataset(test_pos)

# join test and y_label
test_pos = test_pos.join(y_label, on = 'test_id', how = 'left')

# test set contains only positive, labels; suffle to create negative examples
test_neg = test_pos.copy()
test_neg['question1'] = np.random.permutation(test_neg['question1'])
test_neg['is_duplicate'] = 0

# Create final test set
test = pd.concat([test_pos, test_neg], ignore_index = True)

# Reset indexes
train.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)

# tokeniser
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# CV = train.sample(frac = 1).reset_index(drop = True).head(n = 100)
# CV.to_csv("data/SiameseBERT/CV/cv_data.csv", index = False)

CV = pd.read_csv("data/SiameseBERT/CV/cv_data.csv", index_col = False)

CV_dataset = SiameseNetWorkSentenceDataset(data = CV, tokenizer = tokenizer, max_length = 64)
model_params = dict({'freeze_embedding': True, 'freeze_encoder_layer': 8, 'freeze_cls_pooler': True})


Contrastive_m_list = [5, 10, 20]
eval_threshold_list = [1, 5, 10]
batch = 16

for m, th in zip(Contrastive_m_list, eval_threshold_list):
    loss = ConstrastiveLoss(m = m)
    eval_threshold = th
    print("Testing m = " + str(m) + " and th = " + str(th))
    out_dir = os.path.join("data/SiameseBERT", "batch" + str(batch) + "_m" + str(m) + "_th" + str(th))
    print("Output directory: " + out_dir)
    cross_validation(model_params = model_params, dataset = CV_dataset, k = 5, loss_fn = loss, eval_threshold = eval_threshold, device = device, out_dir = out_dir, batch_size = batch, nepochs = 1, save = True, random_state = 1)
