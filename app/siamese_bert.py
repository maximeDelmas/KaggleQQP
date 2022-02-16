import os
import time
from datetime import timedelta
import torch

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from transformers import BertModel
from sklearn.model_selection import KFold

from utils import init_model


class SiameseNetWorkSentenceDataset(Dataset):
    """
    SiameseNetWorkSentenceDataset create a Dataset from a pd.Dataframe
    
    Args:
    - data (pd.DataFrame): the data dataframe with column 'question1' and 'question2' along with the label 'is_duplicate'
    - tokenizer: the BERT tokenizer, such as: BertTokenizer.from_pretrained('bert-base-uncased')
    - max_length: the maximal length of tokens input vector (default 64) Shorter vector arre padded to max_length with [PAD token] (id: 0) and longer are truncated.
    The size includes the start [CLS] and end [SEP] tokens.

    Returns the created Dataset.

    Using the Dataset in a Dataloader:
        dataloader = DataLoader(Dataset, batch_size=N, shuffle=True/False)
        When iterating over the Dataset using a dataloader, specifying the batch size at N, each returned batch is a list of 3 elements:
        - Dataset[0]: dict of input_ids, token_type_ids and attention_mask tensors of size (N x S) for questions 1 in the batch
        - Dataset[1]: dict of input_ids, token_type_ids and attention_mask tensors of size (N x S) for questions 2 in the batch
        - Dataset[2]: tensor corresponding to the label of the questions pairs.

        S is the sequence length.
    """

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        def squeeze_tensors(tks):
            """Take a tensor and remove unnecessary dimension. When using tokenizer with return_tensors = 'pt', the returned tensor is by default 2 dimensions, has it could handle a list of sentence as inputs.
            However, as we only sent one sentence at a time to the tokenizer to create the Dataset, it result in an additional dimension that will be useless after pooling results by batches in the DataLoader

            Args:
                tks ([type]): [description]
            """
            tks.data["input_ids"] = torch.squeeze(tks.data["input_ids"])
            tks.data["token_type_ids"] = torch.squeeze(tks.data["token_type_ids"])
            tks.data["attention_mask"] = torch.squeeze(tks.data["attention_mask"])

        s1 = self.data.loc[index, 'question1']
        s2 = self.data.loc[index, 'question2']
        label = torch.tensor(self.data.loc[index, 'is_duplicate'])

        tokens1 = self.tokenizer(text=s1, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        squeeze_tensors(tokens1)
        tokens2 = self.tokenizer(text=s2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        squeeze_tensors(tokens2)

        return tokens1, tokens2, label



class SiameseBERTNet(nn.Module):
    """SiameseBERTNet

    The SiameBERTNet class is a siamese network where the two sentences are first independently sent to a BERT model.
    The embedding of the sentence is created from the average mean of the token embedding outputed by BERT, [CLS] and [SEP] tokens can be excluded.
    Inputs:
        - input_ids_1: tensor (NxS) of the tokens corresponding ids in WordPiece vocabulary for sentence 1. N is the batch size and S the sequence length (default to 64)
        - attention_mask_1: tensor (NxS) tensor corresponing to the attention mask for sentence 1. Tokens can attent to all other tokens of the sentence 1.
        - input_ids_2: tensor (NxS) of the tokens corresponding ids in WordPiece vocabulary for sentence 2. N is the batch size and S the sequence length (default to 64)
        - attention_mask_2: tensor (NxS) tensor corresponing to the attention mask for sentence 2. Tokens can attent to all other tokens of the sentence 2.

    Args:
        no_cls_pooling (bool): incluse [CLS] token is the sentence averaged embedding ?
        no_sep_pooling (bool): incluse [SEP] token is the sentence averaged embedding ?
        freeze_embedding (bool): freeze BERT's embedding layer ?
        freeze_encoder_layer (int): freeze the BERT first n encoder (None of False if not)
        freeze_cls_pooler (bool): freeze the BERT cls pooler layer ?
    """

    def __init__(self, no_cls_pooling=True, no_sep_pooling=True, freeze_embedding=0, freeze_encoder_layer=False, freeze_cls_pooler=False):
        super(SiameseBERTNet, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        self.no_cls_pooling = no_cls_pooling
        self.no_sep_pooling = no_sep_pooling

        if freeze_embedding:
            self.bert.embeddings.requires_grad_(False)

        # Set the requires_grad attribute of parameters of the first 'freeze_encoder_layer' layers to False. By default there is 12 layers
        for layer in self.bert.encoder.layer[: freeze_encoder_layer]:
            layer.requires_grad_(False)

        if freeze_cls_pooler:
            self.bert.pooler.requires_grad_(False)

    def forward_siamese(self, input_ids, attention_mask):
        """
        From input token ids and attention mask, compute the embedding of the sentence as the average of the tokens over the last hidden state.
        If no_cls_pooling: the [CLS] token is excluded
        If no_sep_pooling: the [SEP] token is excluded

        Args:
            input_ids (tensor): tensor of token input ids
            attention_mask (tensor): attention mask
        Returns:
            avg (tensor): Mean of the last hidden layer vectors
        """

        # Apply BERT and extract last_hidden_state
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state

        # Apply mean pooling on real tokens
        # Make a copy is some changes (CLS or SEP) need to be applied
        pooling_mask = attention_mask.clone()

        # If the CLS output vector should not participate in average pooling
        if self.no_cls_pooling:
            pooling_mask[:, 0] = 0

        # If the SEP output vector should not participate in average pooling
        if self.no_sep_pooling:
            pooling_mask = torch.where(input_ids == 102, 0, pooling_mask)

        # Get mask at the same dimension as last_hidden_state
        expanded_pooling_mask = pooling_mask.unsqueeze(-1)
        expanded_pooling_mask = expanded_pooling_mask.expand(-1, -1, self.hidden_size)

        # Element wise mul between last_hidden_state and mask to then only consider real tokens in the sum
        prod = torch.mul(last_hidden_state, expanded_pooling_mask)

        # Sum all token vectors
        sum_by_tks = torch.sum(prod, dim=1)

        # Get normalisation factor to compute mean
        norm = torch.sum(pooling_mask, dim=-1).unsqueeze(-1)

        # Comptue average
        avg = torch.div(sum_by_tks, norm)

        return avg

# On ne modifie pas la classe Dataset c'est juste en processing des outputs du DataLoader qu'on gèrera l'envoie au modèle
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        forward step of the Siamese network
        """
        out1 = self.forward_siamese(input_ids_1, attention_mask_1)
        out2 = self.forward_siamese(input_ids_2, attention_mask_2)

        return out1, out2


class ContrastiveLoss(nn.Module):
    """Contrastive loss
    Contrastive loss takes the output of the network for a positive example and calculates its distance to an example of the same class and contrasts that with the distance to negative examples.

    Args:
        m (int): the distance from which negative examples are considered sufficiently distants
        p (int): the norm degree (2 for euclidian)
    """
    def __init__(self, m=4, p=2):
        super(ContrastiveLoss, self).__init__()
        self.m = m
        self.p = p
        self.pdist = nn.PairwiseDistance(p=self.p)

    def forward(self, out_q1, out_q2, y):
        """
        forward step of contrastive loss
        """

        D = self.pdist(out_q1, out_q2)
        loss = torch.mean(y * 1/2 * torch.pow(D, 2) + (1 - y) * 1/2 * torch.pow(torch.clamp((self.m - D), min=0), 2))
        return loss


def evalute_siamese(validation_loader, model, loss_fn, threshold, device):
    """Evaluate the performance of the SiameseBERTNet model on the validation dataset.
    This function is specific to the SiameseBERTNet class.

    Args:
        validation_loader (torch.utils.data.dataloader.DataLoader): the validation dataloader
        model (SiameseBERTNet): the SiameseBERT model to eval
        loss_fn (ContrastiveLoss): the initialized constrastive loss function
        threshold (int): the threshold on the distance between 2 sentences. If D < th: duplicated (1), else non_duplicated (0)
        device (torch.device): the device used by torch cpu or gpu.

    Returns:
        [float, float, float]: return the averaged loss, F1-score and accuracy on the validation dataset
    """
    # Put model in test mode
    model.eval()

    epsilon = 1e-7

    v_loss = []
    v_f1 = []
    v_accuracy = []

    # Iterate over validation batches
    for step, batch in enumerate(validation_loader):

        # Get batch data
        v_input_ids_q1 = batch[0]['input_ids'].to(device)
        v_attention_mask_q1 = batch[0]['attention_mask'].to(device)
        v_input_ids_q2 = batch[1]['input_ids'].to(device)
        v_attention_mask_q2 = batch[1]['attention_mask'].to(device)
        v_y = batch[2].to(device)

        # Apply model
        v_out_q1, v_out_q2 = model(v_input_ids_q1, v_attention_mask_q1, v_input_ids_q2, v_attention_mask_q2)

        # Compute Loss
        loss = loss_fn(v_out_q1, v_out_q2, v_y)
        v_loss.append(loss.item())

        # Compute prediction at m
        pwdist = loss_fn.pdist(v_out_q1, v_out_q2)
        pred = torch.where(pwdist < threshold, 1, 0)

        # Compute F1 score:
        tp = (pred * v_y).cpu().numpy().sum()
        precision = tp / (pred.cpu().numpy().sum() + epsilon)
        recall = tp / (v_y.cpu().numpy().sum() + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        v_f1.append(f1)

        # Compute accuracy
        accuracy = (pred == v_y).cpu().numpy().mean() * 100
        v_accuracy.append(accuracy)

    # Compute averaged loss
    avg_loss = np.mean(v_loss)
    avg_f1 = np.mean(v_f1)
    avg_accuracy = np.mean(v_accuracy)

    # return back the model in training mode
    model.train()

    return avg_loss, avg_f1, avg_accuracy




def pwdist_siamese_bert(model, test, device, out_dir):
    """Compute pairwise distances between question pairs in a test or validation set using the model.

    Args:
        model (SiameseBERTNet): the trained model
        test (torch.utils.data.dataloader.DataLoader): the test / validation dataloader
        device (torch.device): the device used by torch cpu or gpu.
        out_dir (str): the output directory
    """

    # set the model to eval:
    model.eval()

    pwdist_list = []
    y_list = []
    # Iterate over testing batchs
    for step, batch in enumerate(test):

        # Get batch data
        input_ids_q1 = batch[0]['input_ids'].to(device)
        attention_mask_q1 = batch[0]['attention_mask'].to(device)
        input_ids_q2 = batch[1]['input_ids'].to(device)
        attention_mask_q2 = batch[1]['attention_mask'].to(device)
        y = batch[2].to(device)

        # Apply model
        out_q1, out_q2 = model(input_ids_q1, attention_mask_q1, input_ids_q2, attention_mask_q2)

        # Compute dist
        pdist = nn.PairwiseDistance(p=2)
        pwdist = pdist(out_q1, out_q2)

        # Concat
        pwdist_list += pwdist.tolist()
        y_list += y.tolist()

    df_pred = pd.DataFrame({'pwdist': pwdist_list, 'y': y_list})
    df_pred.to_csv(os.path.join(out_dir, "best_model_predictions.csv"), index=False)




def train_loop_siamese(model, dataloader, validation, optimizer, scheduler, loss_fn, eval_threshold, nepochs, device, out_dir, save):
    """The training loop for the SiameseBERTNet model. The function exports training logs to evaluate the model performances and overfitting during training.
    The model is trained by minimizing the loss obtained with loss_fn.
    If a validation dataloader is provided the model is evaluate on it using evalute_siamese with 'loss_fn' and 'eval_threshold'. The evaluation is done every 'step' batch in each epoch and also at the end of each epoch. The results are saved in the returned training_logs dataframe.
    The function also saves in the output directory the parameters of the best model obtained during training, based on the minimal evaluation loss that have been obtained (best-model.pt).
    The best model can then be load using:
    model = SiameseBERTNet(**params)
    model.load_state_dict(torch.load("path/to/best-model.pt"))
    If no validation dataloader is provided (None) only the average training loss is reported and the parameters obtained at the end of training will be exported (not necessarily the best)
    This model can be load using the same method as described above.

    Args:
        model (SiameseBERTNet): the SiameseBERTNet model to train
        dataloader (torch.utils.data.dataloader.DataLoader): the training dataLoader
        validation (torch.utils.data.dataloader.DataLoader): the validation dataloader
        optimizer (transformers.AdamW): the parametrized optimizer from init_model
        scheduler (torch.optim.lr_scheduler.LambdaLR): the parametrized scheduler from init_model
        loss_fn (ContrastiveLoss): the initialized constrastive loss function
        eval_threshold (int): the threshold to use for evaluation
        nepochs (int]): the number of training epochs (must be same as in the scheduler)
        device (torch.device): the device used by torch cpu or gpu.
        out_dir (str): the output directory
        save (bool): save the best (or last if no validation) model during training ?

    Returns:
        [pd.DataFrame]: the training_logs dataframe. Reports the average training loss, evaluation loss, F1-score and accuracy computed on the state of the trained model every 'step' batchs in each epoch and also at the end of each epoch.
    """

    # Init errors, F1-score and accuracy vector to store for all epochs
    training_logs = pd.DataFrame()
    _train_errors = []
    _eval_errors = []
    _eval_F1 = []
    _eval_acc = []
    _type = []

    # Init evaluation measure for choosing the best model
    best_model_eval_loss = np.inf

    # Evaluation without training
    if validation:
        avg_validation_loss, avg_f1_validation, avg_acc_validation = evalute_siamese(validation, model, loss_fn, eval_threshold, device)
        _eval_errors.append(avg_validation_loss)
        _eval_F1.append(avg_f1_validation)
        _eval_acc.append(avg_acc_validation)
        _train_errors.append(np.nan)
        _type.append('init')

        # set the best model measure to avg_validation_loss
        best_model_eval_loss = avg_validation_loss
        print("Initial validation loss: " + str(best_model_eval_loss) + " | Initial F1-score: " + str(avg_f1_validation) + " | Initial Accuracy: " + str(avg_acc_validation))

    # Training on epochs
    for i_epoch in range(nepochs):
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # Init
        total_step_in_dataloader = len(dataloader)
        epoch_time, batch_time = time.time(), time.time()
        total_loss, batch_loss, batch_count = 0, 0, 0

        # Put model in train mode (important if a run on the validation in eval mode have been done previously)
        model.train()

        # iterate over batches:
        for step, batch in enumerate(dataloader):

            batch_count += 1

            # Get batch data
            input_ids_q1 = batch[0]['input_ids'].to(device)
            attention_mask_q1 = batch[0]['attention_mask'].to(device)
            input_ids_q2 = batch[1]['input_ids'].to(device)
            attention_mask_q2 = batch[1]['attention_mask'].to(device)
            y = batch[2].to(device)

            # Reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
            model.zero_grad()

            # Apply model
            out_q1, out_q2 = model(input_ids_q1, attention_mask_q1, input_ids_q2, attention_mask_q2)

            # Compute Constrastive loss
            loss = loss_fn(out_q1, out_q2, y)

            # Update batch loss and total loss
            total_loss += loss.item()
            batch_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Checking every 10 steps:
            if ((((step + 1) % 25) == 0) and step != 0) or step == (total_step_in_dataloader - 1):
                time_elapsed = str(timedelta(seconds=(time.time() - batch_time)))
                batch_avg_loss = batch_loss/batch_count
                _train_errors.append(batch_avg_loss)

                if validation:
                    avg_validation_loss, avg_f1_validation, avg_acc_validation = evalute_siamese(validation, model, loss_fn, eval_threshold, device)
                    _eval_errors.append(avg_validation_loss)
                    _eval_F1.append(avg_f1_validation)
                    _eval_acc.append(avg_acc_validation)

                    # Print summary
                    print(f"batch {step+1:>6d} / {total_step_in_dataloader:>4d} | Elapsed {time_elapsed} | Average loss on the previous {batch_count:>4d} batchs : {batch_avg_loss:5.2f} | Average validation loss: {avg_validation_loss:6.2f} | Average F1-score: {avg_f1_validation:6.2f} | Average Accuracy: {avg_acc_validation:6.2f} %")
                else:
                    print(f"batch {step+1:>6d} / {total_step_in_dataloader:>4d} | Elapsed {time_elapsed} | Average loss on the previous {batch_count:>4d} batchs : {batch_avg_loss:5.2f} | ")

                _type.append(False)

                # Reset batch_count, batch_loss and batch_time
                batch_loss, batch_count = 0, 0
                batch_time = time.time()

        avg_train_loss_epoch = total_loss/total_step_in_dataloader
        _train_errors.append(avg_train_loss_epoch)

        time_elapsed_epoch = str(timedelta(seconds=(time.time() - epoch_time)))
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        # Test current model ( at epcoch i ) on validation
        if validation:
            avg_validation_loss, avg_f1_validation, avg_acc_validation = evalute_siamese(validation, model, loss_fn, eval_threshold, device)
            _eval_errors.append(avg_validation_loss)
            _eval_F1.append(avg_f1_validation)
            _eval_acc.append(avg_acc_validation)

            # At the end of each epoch, check is the model has a better average evaluation loss than the previous best model:
            if save and (avg_validation_loss < best_model_eval_loss):

                print("\n/!\ save new best model /!\ \n")
                torch.save(model.state_dict(), os.path.join(out_dir, "best-model.pt"))

                # Save best epoch
                with open(os.path.join(out_dir, "epoch.log"), 'w') as f:
                    f.write("best model at epoch: " + str(i_epoch+1))

                best_model_eval_loss = avg_validation_loss

            print(f"Epoch {i_epoch+1:>6d} / {nepochs:>4d} | Elapsed {time_elapsed_epoch} | Average loss on epoch: {avg_train_loss_epoch:18.2f} | Average validation loss: {avg_validation_loss:6.2f} | Average F1-score: {avg_f1_validation:6.2f} | Average Accuracy: {avg_acc_validation:6.2f} %")
        else:
            print(f"Epoch {i_epoch+1:>6d} / {nepochs:>4d} | Elapsed {time_elapsed_epoch} | Average loss on epoch: {avg_train_loss_epoch:18.2f} |")

        _type.append(True)

    if validation:
        training_logs = pd.DataFrame({"Type": _type, "Training.loss": _train_errors, "Eval.loss": _eval_errors, "Eval.F1": _eval_F1, "Eval.Acc": _eval_acc})

    else:
        training_logs = pd.DataFrame({"Type": _type, "Training.loss": _train_errors})

        # if no validation was used, we just save the model at the end of the training
        print("\nsave last model")
        if save:
            torch.save(model.state_dict(), os.path.join(out_dir, "last-model.pt"))

    return training_logs


def cross_validation(model_params, dataset, k, loss_fn, eval_threshold, device, out_dir, batch_size=8, nepochs=4, save=True, random_state=None):
    """The function performs a cross-validation (CV) loop of the training set.
    The training dataset is first split into k fold. For instance setting k to 5 means deviding the training into 5 distinct parts (index attributions is random) and then in each fold we use part as the validation set and the remaining as the training set.
    In each fold we applied the train_loop_siamese. and export the training_logs dataframe in the output directory

    Args:
        model_params (dict): a python dict containing the parameters to initialize a SiameseBERTNet model.
        dataset (torch.utils.data.Dataset): the training dataset that will be used for CV
        k (int): the number of fold
        loss_fn (ContrastiveLoss): the initialized constrastive loss function
        eval_threshold (int): the threshold to use for evaluation
        device (torch.device): the torch device cpu or gpu
        out_dir (str): the output directory
        batch_size (int, optional): the batch size
        nepochs (int, optional): the number of training epochs (must be same as in the scheduler)
        save (bool, optional): save the best model at each fold ?
        random_state(int , optional): set the seed for random_state of KFold (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
    """

    # Init kfold: split the dataset into k folds. shuffle = True indicates that the individuals of the different folds are chosen randomly and are not necesseraly packs that follow each others
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Loop over folds: At each steps, (k - 1) folds are chosen to be in the training set and the remaining kième fold is chose to be the validation set.
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset)):

        # Check outdir
        fold_out_dir = os.path.join(out_dir, "f" + str(fold + 1))
        if not os.path.isdir(fold_out_dir):
            os.makedirs(fold_out_dir)

        print(" --- fold: " + str(fold + 1) + " --- ")
        # To use the index of the individuals belonging to the (k - 1) training folds and the validation fold in the DataLoader, we create SubsetRandomSampler
        # It creates a random sampler with the index in the (k - 1) training folds and the validation fold
        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)

        # We then create 2 data loader: one for iterative with batches over the train_ids and the second for the validation_ids
        # We don't need to use shuffle in the DataLoader as the selection of the ids is done with the provided samplers train_subsampler and validation_subsampler
        train_loader = DataLoader(dataset, batch_size, sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size, sampler=validation_subsampler)

        # Now, train the model with the train_loader and evaludate it on the validation_loader
        model = SiameseBERTNet(**model_params)

        # Check the parameters to be fine-tune
        # check_params(model)

        # Check for cuda:
        model.to(device)
        optimizer, scheduler = init_model(model, train_loader, nepochs)
        k_training_logs = train_loop_siamese(model, train_loader, validation_loader, optimizer, scheduler, loss_fn, eval_threshold, nepochs, device, fold_out_dir, save)
        k_training_logs.to_csv(os.path.join(fold_out_dir, "k_" + str(fold + 1) + "_training_logs.csv"), index=True, header=True)

        # reload the best model and make prediction on validation set

        if os.path.isfile(os.path.join(fold_out_dir, "best-model.pt")):
            print("- Reload and compute prediction on best model")
            model = SiameseBERTNet(**model_params)
            model.load_state_dict(torch.load(os.path.join(fold_out_dir, "best-model.pt")))
            model.to(device)
            pwdist_siamese_bert(model, validation_loader, device, fold_out_dir)
        else:
            print("No best model has been found, skip evaluation on best model")
        