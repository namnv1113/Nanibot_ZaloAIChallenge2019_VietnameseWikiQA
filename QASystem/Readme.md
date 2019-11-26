# Question Answering System

## How to run
This system required a **multilingual pre-training model**: the *multi_cased* (BERT-Base, Multilingual Cased) model for Vietnamese NLP tasks, which are stored in the `model` folder. Note that I didn't include the pretraining code for the BERT model, but if you want to pre-train your model, you can follow the instructions in this [link](https://github.com/google-research/bert#pre-training-with-bert). The model can be download in the [BERT github page](https://github.com/google-research/bert#pre-trained-models). 


To run the model:
```sh
BERT_BASE_PATH='./model/multi_cased'
DATASET_PATH='../dataset/'
OUT_DIR='./finetuned/classifier/'

python run_zalo.py \
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
- `--train_filename` The name of the training file that is stored in the dataset folder (Default is *train.json*)
- `--dev_filename` The name of the development file that is stored in the dataset folder (Default is *None*)
- `--test_filename` The name of the training file that is stored in the dataset folder (Default is *test.json*)
- `--test_predict_outputmode` The mode in which the predict file should be (can be either Zalo-defined format *`zalo`* or full format *`full`*) (Default is *zalo*) (*zalo* mode mainly used for submission on the Zalo test set & full mode is used for test data insight on a dataset with the same format with training data)
- `--max_sequence_len` The maximum input sequence length for embeddings (Default is *256*)
- `--do_lowercase` Should the input text be lowercased (this should be the same as the `do_lowercase` settings in the BERT pretrained model)
- `--model_learning_rate` The default model learning rate (Default is *1e-5*)
- `--model_batch_size` Training batch size (Default is *16*)
- `--train_epochs` Number of loops to train the whole dataset (Default is *3*)
- `--train_dropout_rate` Default dropout rate (Default is *0.1*)
- `--bert_warmup_proportion` Proportion of training to perform linear learning rate warmup (Default is *0.1*)
- `--use_pooled_output` Set to *True* if use pooled output for pretrained BERT output (or fully connected layer input). Set to *False* to use meaned output instead (Default is *True*) 
- `--loss_type` The default loss function to use when training (Can be *cross_entropy*, *focal_loss*, *hinge*, *squared_hinge* or *kld*) (default is *cross_entropy*)
- `--loss_label_smooth` A hyperparameter for doing label smoothing when calculating loss (in [0, 1]). When 0, no smoothing occurs. When positive, the binary ground truth labels `y_true` are squeezed toward 0.5, with larger values of `label_smoothing` leading to label values closer to 0.5. (Default is *0*)
- `--save_checkpoint_steps` The number of steps between each checkpoint save (Default is *500*)
- `--save_summary_steps` The number of steps between each summary write (Default is *100*)
- `--keep_checkpoint_max` The maximum number of checkpoints to keep (Default is *1*)
- `--encoding` The default encoding used in the training dataset (Default set to *utf-8*)
- `--zalo_predict_csv_file` Destination for the Zalo submission predict file during *predict_test* (Default is *./zalo.csv*)
- `--eval_predict_csv_file` Destination for the development set predict file during *train* and *eval* (Default is *None*)
- `--dev_size` The size of the development set taken from the training set. If dev_filename exists, this is ignored. (Default is *0.2*)
