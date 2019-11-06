# Question Answering System

## Requirements
	* Python 3.x (Tested on Python 3.6.7)
	* flask
	* tensorflow==1.11 (or tensorflow-gpu==1.11)

## How to run
This system required a **multilingual pre-training model**: the *multi_cased* (BERT-Base, Multilingual Cased) model for Vietnamese NLP tasks, which are stored in the `model` folder. Note that I didn't include the pretraining code for the BERT model, but if you want to pre-train your model, you can follow the instructions in this [link](https://github.com/google-research/bert#pre-training-with-bert). The model can be download in the [BERT github page](https://github.com/google-research/bert#pre-trained-models). 


To run the model:
```sh
BERT_BASE_PATH='./model/multi_cased'
DATASET_PATH='../dataset/'
OUT_DIR='./finetuned/classifier/'

!python run_zalo.py \
    --mode [train/eval/predict_test/predict_manual] \
    --dataset_path $DATASET_PATH \
    --bert_model_path $BERT_BASE_PATH \
    --model_path $OUT_DIR \
```

Required parameters:
- `--mode` Which mode to you want to run the model (*'train'* for training, *'eval'* for development set evaluation, *'predict_test'* for test set predicting & *'predict_manual'* for manual testing)
- `--dataset_path` The path directory that store the required dataset (note that *train.json* & *test.json* with Zalo format or its preprocessed tfrecords file must be contained in that folder)
- `--bert_model_path` The path to the pretrained BERT model
- `--model_path` The location where the fine-tuned model should be stored

Optional parameters
- `--max_sequence_len` The maximum input sequence length for embeddings (Default is *384*)
- `--do_lowercase` Should the input text be lowercased (this should be the same as the `do_lowercase` settings in the BERT pretrained model)
- `--model_learning_rate` The default model learning rate (Default is *2e-5*)
- `--model_batch_size` Training batch size (Default is *8*)
- `--train_epochs` Number of loops to train the whole dataset (Default is *1*)
- `--train_dropout_rate` Default dropout rate (Default is *0.1*)
- `--bert_warmup_proportion` Proportion of training to perform linear learning rate warmup (Default is *0.1*)
- `--l2_regularization_lambda` Constant for L2 regularization (Default is *0.01*)
- `--save_checkpoint_steps` The number of steps between each checkpoint save (Default is *500*)
- `--save_summary_steps` The number of steps between each summary write (Default is *100*)
- `--keep_checkpoint_max` The maximum number of checkpoints to keep (Default is *1*)
- `--encoding` The default encoding used in the training dataset (Default set to *utf-8*)
- `--zalo_predict_csv_file` Destination for the Zalo submission predict file during *predict_test* (Default is *./zalo.csv*)
- `--eval_predict_csv_file` Destination for the development set predict file durting *train* and *eval* (Default is *None*)
- `--dev_size` The size of the development set taken from the training set (Default is *0.2*)
- `--force_data_balance` Balance training data by truncate training instance whose label is overwhelming (Default is *False*)
    
