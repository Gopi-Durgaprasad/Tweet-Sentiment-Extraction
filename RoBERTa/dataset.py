import torch


def find_start_and_end(tweet, selected_text):
    len_st = len(selected_text)
    start = None
    end = None
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind: ind+len_st] == selected_text:
            start = ind
            end = ind + len_st - 1
            break
    return start, end

def process_with_offsets(args, tweet, selected_text, sentiment, tokenizer):
    
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    

    start_index, end_index = find_start_and_end(tweet, selected_text)

    char_targets = [0]*len(tweet)
    if start_index != None and end_index != None:
        for ct in range(start_index, end_index+1):
            char_targets[ct] = 1
    
    encoded = tokenizer.encode_plus(
                    f"{sentiment} words",
                    tweet,
                    max_length=args.max_seq_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    return_offsets_mapping=True
                )

    target_idx = []
    for j, (offset1, offset2) in enumerate(encoded["offset_mapping"]):
        if j > 4:
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

    encoded["start_position"] = target_idx[0]
    encoded["end_position"] = target_idx[-1]
    encoded["tweet"] = tweet
    encoded["selected_text"] = selected_text
    encoded["sentiment"] = sentiment

    sentiment_class = {
        "positive" : 0,
        "negative" : 1,
        "neutral" : 2
    }

    encoded["sentiment_label"] = sentiment_class[sentiment]

    return encoded


class TweetDataset:
    def __init__(self, args, tokenizer, df, mode="train", fold=0):
        
        self.mode = mode

        if self.mode == "train":
            df = df[~df.kfold.isin([fold])].dropna()
            self.tweet = df.text.values
            self.sentiment = df.sentiment.values
            self.selected_text = df.selected_text.values
        
        elif self.mode == "valid":
            df = df[df.kfold.isin([fold])].dropna()
            self.tweet = df.text.values
            self.sentiment = df.sentiment.values
            self.selected_text = df.selected_text.values
        
        self.tokenizer = tokenizer
        self.args = args
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):

        tweet = str(self.tweet[item])
        selected_text = str(self.selected_text[item])
        sentiment = str(self.sentiment[item])
        
        features = process_with_offsets(
                        args=self.args, 
                        tweet=tweet, 
                        selected_text=selected_text, 
                        sentiment=sentiment, 
                        tokenizer=self.tokenizer
                    )
        
        return {
            "input_ids":torch.tensor(features["input_ids"], dtype=torch.long),
            "token_type_ids":torch.tensor(features["token_type_ids"], dtype=torch.long),
            "attention_mask":torch.tensor(features["attention_mask"], dtype=torch.long),
            "start_position":torch.tensor(features["start_position"],dtype=torch.long),
            "end_position":torch.tensor(features["end_position"], dtype=torch.long),

            "sentiment_label":torch.tensor(features["sentiment_label"], dtype=torch.long),
            

            "offsets":features["offset_mapping"],
            "tweet":features["tweet"],
            "selected_text":features["selected_text"],
            "sentiment":features["sentiment"]
        }