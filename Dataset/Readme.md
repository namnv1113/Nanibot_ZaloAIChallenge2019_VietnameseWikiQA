This folder contains training and testing data for the project with [SQuAD-like format](https://rajpurkar.github.io/SQuAD-explorer/).

## General
The training and testing data include: Vietnamese question-answer pairs from Wikipedia (same as SQuAD) and Vietnamese question-answer pairs from UIT regulation documents.

Type of questions varies from factoid questions to questions that require world knowledge, questions with syntactic/lexical variations, questions require multiple sentences reasoning and questions with a long answers.  

## Content
- **Train** dataset
	- **train_reg.json**: include hand-craft qa pairs from the UIT regulation documents
	- **train_wiki.json**: include hand-craft qa pairs from the Vietnamese Wikipedia.
	- **train.json**: include the combined train_reg.json and train_wiki.json
	- **train_translated_squad.json**: include the translated training SQuAD dataset from English to Vietnamese
	- **train_translated_squad_25.json**: include 25% best translation of the translated SQuAD
	- **train_translated_squad_50.json**: include 50% best translation of the translated SQuAD
	- **train_translated_squad_75.json**: include 75% best translation of the translated SQuAD
	- **train_translated_squad_100.json**: include 100% best translation of the translated SQuAD
- **Test** dataset
	- **test_reg.json**: include hand-craft qa pairs from the UIT regulation documents
	- **test_wiki.json**: include hand-craft qa pairs from the Vietnamese Wikipedia.
	- **test.json**: include the combined test_reg.json and test_wiki.json


|		Training set							|	QA pairs	|	Sentence count	|
|-----------------------------------------------|--------------:|------------------:|
|Train_Wikipedia_Handcraft (train_wiki): 	 	|	  599		|		  67		|
|Train_UITReg_Handcraft (train_reg):			|	  626		|		  75		|
|Handcraft (combined)							|	  1225		|		 142		|
|Train_TranslatedSquad (train_translated_squad)	|	 78978      |					|
	
|		Testing set								|	QA pairs	|	Sentence count	|
|-----------------------------------------------|--------------:|------------------:|
|Test (Wikipedia + UITReg)						|	  243		|		  37		|
|	test_reg									|      95	    |		  13		|
|	test_wiki									|     111		|		  14		|