import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, model_path, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(model_path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 4, 2) #768
        #self.l1 = nn.Linear(768 * 4, 1)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        #self.init_weights()


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        _,_, out = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2], out[-3], out[-4]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

class TweetModel2(transformers.BertPreTrainedModel):
    def __init__(self, model_path, conf):
        super(TweetModel2, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(model_path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 7, 2) #768
        #self.l1 = nn.Linear(768 * 4, 1)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        #self.init_weights()


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        out1,out2, out = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        #print(out1.shape)
        #print(out2.shape)
        #print(len(out))
        #print(out)

        hidden_list = list(out)
        hidden_stack = torch.stack(hidden_list, dim=0)

        SUM = torch.sum(hidden_stack, dim=0)
        MEAN = torch.mean(hidden_stack, dim=0)
        #MAX = torch.max(hidden_stack, dim=0)
        #MIN = torch.min(hidden_stack, dim=0)
        #MEAN = torch.stack(hidden_list, dim=
        #print(SUM.shape, MEAN.shape, MAX, MIN)


        out = torch.cat((out[-1], out[-2], out[-3], out[-4], out1, SUM, MEAN), dim=-1)
        #print(out.shape)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        #print(start_logits.shape, end_logits.shape)

        return start_logits, end_logits

class TweetModelSentiment(transformers.BertPreTrainedModel):
    def __init__(self, model_path, conf):
        super(TweetModelSentiment, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(model_path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 4, 2) #768
        self.l1 = nn.Linear(768, 3)
        #self.l1 = nn.Linear(768 * 4, 1)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        #self.init_weights()


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        _,o1, out = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2], out[-3], out[-4]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        o1 = self.drop_out(o1)
        o1 = self.l1(o1)

        return start_logits, end_logits, o1
