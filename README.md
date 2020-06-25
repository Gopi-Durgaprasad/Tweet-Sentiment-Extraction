# Tweet-Sentiment-Extraction
Extract support phrases for sentiment labels

### Public LeaderBord 179/2227
### Private LeaderBord 592/2227

## Discription

"My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

## Problem Statement

You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.)

## Data

- `train.csv` - the training set
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format

### Columns

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

## Experments

- In this competiton we are using TPU's for traing all kind of transformer models.
- we achive best results with `RoBERTa` model

## Results

| Model          | CV    | Public LB | Pravete LB|
| :------------  |:-----:| -----:    | -----:    |
| Bert-base      | 0.651 | 0.669     | 0.668     |
| Roberta-base   | 0.705 | 0.716     | 0.716     |
| Albert-xxlarge | 0.701 | 0.694     | 0.704     |
| XLM-RoBERTa    | 0.692 |   -       |    -      |
| XLNet          | 0.654 |   -       |    -      |    



