import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def jaccard(str1, str2):
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): 
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def calculate_jaccard_score(features_dict, start_logits, end_logits, tokenizer):

    binput_ids = to_list(features_dict["input_ids"])
    btweet = features_dict["tweet"]
    bselected_text = features_dict["selected_text"]
    bsentiment = features_dict["sentiment"]
    boffsets = features_dict["offsets"]

    bstart_logits = np.argmax(F.softmax(start_logits, dim=1).cpu().data.numpy(), axis=1)
    bend_logits = np.argmax(F.softmax(end_logits, dim=1).cpu().data.numpy(), axis=1)

    jac_list = []

    for i in range(len(btweet)):

        idx_start = bstart_logits[i]
        idx_end = bend_logits[i]
        offsets = boffsets[i]
        input_ids = binput_ids[i]
        tweet = btweet[i]
        selected_text = bselected_text[i]

        if idx_end < idx_start:
            idx_end = idx_start

        filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1], skip_special_tokens=True)

        if bsentiment[i] == "neutral" or len(tweet.split()) < 2:
            filtered_output = tweet
        
        jac = jaccard(selected_text.strip(), filtered_output.strip())

        jac_list.append(jac)

    return np.mean(jac_list)