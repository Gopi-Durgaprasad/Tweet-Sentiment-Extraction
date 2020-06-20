import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class TweetModel(transformers.XLNetPreTrainedModel):
    def __init__(self, model_path, conf):
        super(TweetModel, self).__init__(conf)
        self.xlnet = transformers.XLNetModel.from_pretrained(model_path, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 4, 2) #762
        #self.l1 = nn.Linear(768 * 4, 1)
        #torch.nn.init.normal_(self.l0.weight, std=0.02)
        #self.init_weights()


    def forward(self,input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        _, out = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask
        )

        out = torch.cat((out[-1], out[-2], out[-3], out[-4]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits