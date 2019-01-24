# Question Answering System for Regulations of University of Information Technology

## General

The purpose of this project is to develop a *Question Answering System* with *Reading Comprehension* ability on Vietnamese, whose tools and resources are lacked, and applied to answering question related to rules and regulations of University of Information System.

This system adapts traditional *Information Retrieval* techniques (mostly based on *Extended Boolean Model*) and *Deep Learning* algorithms ([*BERT*](https://arxiv.org/pdf/1810.04805.pdf) that achieves state-of-the-art performance on 11 different NLP tasks in English) and *transfer learning* on Vietnamese that *posed attractive potential* on Vietnamese Question Answering researches. 

The Information Retrieval aprroaches are very common, but deep learning approaches are almost never used in Vietnamese QA System. In this project, a naive transfer learning technique is used, where we translate the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) from English to Vietnamese and remove bad translation ([link](http://www.lrec-conf.org/proceedings/lrec2018/pdf/711.pdf)) that poses an additional 10% boost in F1 accuracy, resulted in an F1 accuracy of 66% in the original task (QA on Wikipedia) and 56% in the UIT regulations task.

More information about this project is stored in ./Report/Summary.pdf or ./Report/Thesis.pdf

## Structures
* *QASystem* and *Ultilities* contain source codes, base model as well as fine-tuning models and dataset used in this project. Guide on how to setup and re-produce the result is also provided.
* *Report* contains documents about this thesis as well as slides and related files.
* *Dataset* contains the dataset that is used in this project.

## Information
* By Nguyễn Việt Nam - 14520560
* Advisors: Dr. Ngô Đức Thành & M Sc Nguyễn Vinh Tiệp
* Advanced Education Program 2014 - VNU-UIT

If any problem occurs, please contact me via my email address 14520560@gm.uit.edu.vn or namnv1113@gmail.com