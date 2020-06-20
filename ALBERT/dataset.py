import torch
import sentencepiece_pb2
import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        token_ids = []
        tokens = []
        for piece in spt.pieces:
            token_ids.append(piece.id)
            offsets.append((piece.begin, piece.end))
            tokens.append(piece.surface)
        return token_ids, offsets, tokens


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

def create_offsets(tokens):
    offsets = []
    off1 = 0
    off2 = 0
    first = True
    for token in tokens:
        if '▁' in token:
            ntoken = token.replace('▁','')
            if not first:
                off1 = off1 + 1
            off2 = off1+len(ntoken)
            offsets.append((off1,off2))
            off1 = off2  
            first = False
        else:
            off2 = off1+len(token)
            offsets.append((off1,off2))
            off1 = off2
            
    return offsets

def process_with_offsets(args, tweet, selected_text, sentiment, tokenizer, sptokenizer):

    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    start_index, end_index = find_start_and_end(tweet, selected_text)

    char_targets = [0]*len(tweet)
    if start_index != None and end_index != None:
        for ct in range(start_index, end_index+1):
            char_targets[ct] = 1
    
    encoded = tokenizer.encode_plus(
                    sentiment,
                    tweet,
                    max_length=args.max_seq_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    #return_offsets_mapping=True
                )
    
    #_, offsets, _ = sptokenizer.encode(tweet)
    tweet_tokens = tokenizer.tokenize(tweet)
    offsets = create_offsets(tweet_tokens)
    tweet_offsets = [(0,0)]*3 + create_offsets(tweet_tokens) + [(0,0)]

    padding_length = args.max_seq_len - len(tweet_offsets)
    if padding_length > 0:
        tweet_offsets = tweet_offsets + ([(0,0)] * padding_length)
    
    encoded["offset_mapping"] = tweet_offsets

    target_idx = []
    for j, (offset1, offset2) in enumerate(encoded["offset_mapping"]):
        if j > 2:
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

    encoded["start_position"] = target_idx[0]
    encoded["end_position"] = target_idx[-1]
    encoded["tweet"] = tweet
    encoded["selected_text"] = selected_text
    encoded["sentiment"] = sentiment

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
        self.sptokenizer = SentencePieceTokenizer(args.sp_path)
    
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
                        tokenizer=self.tokenizer,
                        sptokenizer=self.sptokenizer
                    )
        
        return {
            "input_ids":torch.tensor(features["input_ids"], dtype=torch.long),
            "token_type_ids":torch.tensor(features["token_type_ids"], dtype=torch.long),
            "attention_mask":torch.tensor(features["attention_mask"], dtype=torch.long),
            "start_position":torch.tensor(features["start_position"],dtype=torch.long),
            "end_position":torch.tensor(features["end_position"], dtype=torch.long),

            "offsets":torch.tensor(features["offset_mapping"], dtype=torch.long),
            "tweet":features["tweet"],
            "selected_text":features["selected_text"],
            "sentiment":features["sentiment"]
        }