This folder contains training and testing data for the project with [SQuAD-like format](https://rajpurkar.github.io/SQuAD-explorer/).

## General
The training and testing data include: Vietnamese question-answer pairs from Wikipedia (same as SQuAD)
Type of questions varies from factoid questions to questions that require world knowledge, questions with syntactic/lexical variations, questions require multiple sentences reasoning and questions with a long answers.  

## Content
- **Train** dataset
	- **train_translated_squad.json**: include the translated training SQuAD dataset from English to Vietnamese
	- **train_translated_squad_25.json**: include 25% best translation of the translated SQuAD
	- **train_translated_squad_50.json**: include 50% best translation of the translated SQuAD
	- **train_translated_squad_75.json**: include 75% best translation of the translated SQuAD

|		Training set							|	QA pairs	|	Sentence count	|
|-----------------------------------------------|--------------:|------------------:|
|Train_TranslatedSquad (train_translated_squad)	|	 78978      |					|
	
|		Testing set								|	QA pairs	|	Sentence count	|
|-----------------------------------------------|--------------:|------------------:|
|Test 											|	  0			|		  0			|