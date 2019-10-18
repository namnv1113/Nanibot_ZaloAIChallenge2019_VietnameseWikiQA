# Zalo AI Challenge 2019 - Vietnamese Wikipedia Question Answering

## General
This repository represents the works of the Nanibot Team on the **Vietnamese Wikipedia Question Answering** task on the **Zalo AI Challenge 2019**.

The works on this repository is based on a previous work on a similar task (Question Answering for regulations of UIT), which is available [here](https://github.com/phateopera/UITHelper_QAS) 

## Structures
* *QASystem* and *Ultilities* contain source codes, base model as well as fine-tuning models and dataset used in this project. Guide on how to setup and re-produce the result is also provided.
* *Dataset* contains the dataset that is used in this project.

## Team Members
* Nguyễn Việt Nam (namnv1113@gmail.com)
* Trần Trí Nguyên (nguyen.tran.forte1609@gmail.com)
* Nguyễn Minh Hiếu (hieunguyenvn98@gmail.com)

## How to run
To run the demonstration:
```sh
python demo.py 
```
or 
```sh
export FLASK_APP=demo.py
flask run
```

*Note: See this [readme](/QASystem/Readme.md) for information on how to setup before running*