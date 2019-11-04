## General
This folder contains training and testing data for the Zalo AI Challenge 2019.

## Content
- **Train** dataset
    - **train.json**: The training dataset provided by Zalo (data format is defined [here](https://challenge.zalo.ai/portal/question-answering/data)). The dataset is in Vietnamese.
    - **squad-train-v2.0.json** &  **squad-dev-v2.0.json** : The training & development dataset for the [SQuAD question answering task](https://rajpurkar.github.io/SQuAD-explorer/). The dataset is in English.
	- **train_translated_squad.json**: Include the translated training SQuAD dataset from English to Vietnamese (note that currently this file contains training data the SQuAD v1.1 only).
- **Test** dataset
    - **test.json**: The test dataset provided by Zalo (data format is defined [here](https://challenge.zalo.ai/portal/question-answering/data)). Used for leaderboard scoring. The dataset is in Vietnamese.

|		Training set							|	QA pairs	|
|-----------------------------------------------|--------------:|
|   (train.json)	                            |	 18108      |
|   (squad-train-v2.0.json.json)	            |	130319      |
|   (squad-dev-v2.0.json.json)	                |	 11873      |
|   (train_translated_squad.json)	            |	 78978      |
	
|		Testing set								|	QA pairs	|	
|-----------------------------------------------|--------------:|
|   (test.json)									|	   501		|