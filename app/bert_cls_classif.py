import torch
from transformers import BertModel
from torch.utils.data import Dataset
from torch import nn



class BERTSentencesClassificationDataset(Dataset):
    """Dataset creater for BERTSentencesClassification

    Args:
    SiameseNetWorkSentenceDataset create a Dataset
    - data (pd.DataFrame): the data dataframe with column 'question1' and 'question2' along with the label 'is_duplicate'
    - tokenizer: the BERT tokenizer, such as: BertTokenizer.from_pretrained('bert-base-uncased')
    - max_length: the maximal length of tokens input vector (default 64) Shorter vector arre padded to max_length with [PAD token] (id: 0) and longer are truncated.
    The size includes the start [CLS] and end [SEP] tokens.
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

        t = self.tokenizer(s1, s2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        squeeze_tensors(t)

        label = torch.tensor(self.data.loc[index, 'is_duplicate'])

        return t, label



class BERTSentencesClassification(nn.Module):
    """BERTSentencesClassification model
    """

    def __init__(self):
        super(BERTSentencesClassification, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        self.out = 1
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.hidden_size, out_features=self.out, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        """forwars function
        """

        # Get input_ids, token_type_ids (as we have sentense pairs) and attention mask
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = out.pooler_output
        classification = self.classifier(cls_token)

        return classification, cls_token
    