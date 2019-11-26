# Zalo AI Challenge 2019 - Vietnamese Wikipedia Question Answering

## General
This repository represents the works of the Nanibot Team on the **Vietnamese Wikipedia Question Answering** task on the **Zalo AI Challenge 2019**.

The works on this repository is based on a previous work on a similar task ([Question Answering for regulations of UIT](https://github.com/phateopera/UITHelper_QAS))

## Structures
* *QASystem* and *Ultilities* contain source codes, base model as well as fine-tuning models and dataset used in this project. Guide on how to setup and re-produce the result is also provided.
* *Dataset* contains the dataset that is used in this project.

## Team Members
* [Nguyễn Việt Nam](https://github.com/phateopera)
* [Trần Trí Nguyên](https://github.com/nguyentranforte1609)
* [Nguyễn Minh Hiếu](https://github.com/hieudepchai)
* [Nguyễn Trường Phát](https://github.com/patrickphatnguyen)

## How to run
Details on how to train/predict using the model is described [here](https://github.com/phateopera/Nanibot_ZaloAIChallenge2019_VietnameseWikiQA/blob/master/QASystem/Readme.md)


## What we have tried

- [x] Apply BERT as baseline for the [QA problem defined by Zalo](https://challenge.zalo.ai/portal/question-answering)
- [x] Data augmented using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) by translating & [de-noising](www.lrec-conf.org/proceedings/lrec2018/pdf/711.pdf)
- [x] Improve BERT by trying different approaches ([BERT + TextCNN](https://github.com/phateopera/Nanibot_ZaloAIChallenge2019_VietnameseWikiQA/tree/bert_and_textcnn), [BERT with additional fully-connected layer](https://github.com/phateopera/Nanibot_ZaloAIChallenge2019_VietnameseWikiQA/tree/add_fc1)), but yeild no improvements
- [x] Try different loss function for the classification problem ((Squared) Hinge loss, KLD loss & Focal loss) along with label smoothing, but yeild no improvements
- [ ] Data augmented using backtranslation
- [ ] Apply [multilligual RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) for the problem

Our solution yeild an F1 score of *79.15%*, ranked *11* in the public leaderboard of the Zalo AI Challenge 2019 for the Vietnamese Wiki Question Answering problem for the public test set.
