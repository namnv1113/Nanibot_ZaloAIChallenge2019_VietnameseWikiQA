# Question Answering System

## Requirements
	* Python 3.x (Tested on Python 3.6.7)
	* whoosh
	* pyvi
	* gensim
	* flask
	* flask-session
	* tensorflow==1.11 (or tensorflow-gpu==1.11)

This system required 2 pre-training models: the *base* model for English BERT demonstration and the *multi_cased* model for Vietnamese tasks, which are stored in the *model* folder. Note that I didn't include the pretraining code for the BERT model, but if you want to pre-train your model, you can follow the instructions in this [link](https://github.com/google-research/bert#pre-training-with-bert). The model can be download in the [BERT github page](https://github.com/google-research/bert#pre-trained-models). 

Beside, a word2vec model is required for query expansion in the search engine, which can be downloaded using this [link](https://drive.google.com/file/d/1R-Qq6RrlXCZpPhnMirxsPhycTiPfH09m/view?usp=sharing), or another one can be used. If so, please rename the vi_wordvectors_file variable in the source code ulti.py

3 fine-tuned models are used: the *eng* model for English machine comprehension task SQuAD, the *vi_uit* model for Vietnamese reading comprehension task on UIT regulations, and the *vi* model for Vietnamese reading comprehension task on SQuAD. Since these 3 files are too large to stored in the CD, please download the fine-tuned model from this [link](https://drive.google.com/drive/folders/1g-9IJdYlelUSR2DHh9AEqVnPfqddoIhs?usp=sharing) and put them in the *finetuned* folder.

To run the demonstration:
```sh
python demo.py 
```
or 
```sh
export FLASK_APP=demo.py
flask run
```

To create/update the index the documents (serve as a knowledge base for UIT regulations QA), store/update the documents in a folder (*texts* folder is served as an example for UIT regulations corpus, can be downloaded here https://drive.google.com/drive/folders/19NDrVByaoU3qAW4no72UGTiBS4y51WKg?usp=sharing), then run the command below to update the index.
```sh
python whoosh_index.py -m create -f <folder_contains_document>
```
or
```sh
python whoosh_index.py -m update
```

To run the fine-tuning or predicting:
```sh
BERT_BASE_DIR='./model/multi_cased'
SQUAD_DIR='./squad/squad_vi'
OUT_DIR='./finetuned/vi/'

!python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --do_predict=True \
  --train_file=$SQUAD_DIR/<train_file> \
  --predict_file=$SQUAD_DIR/<test_file> \
  --do_lower_case=False \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=5.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_answer_length=500 \
  --output_dir=$OUT_DIR
```
Where 
* *BERT_BASE_DIR* is the folder contains the pre-trained model (stored in the *model* folder)
* *SQUAD_DIR* is the folder contains the dataset (stored in the *dataset* folder) ()
* *OUT_DIR* is the the location to output the fine-tuned model

Include this if TPU is used
```sh
   --use_tpu=True \
   --tpu_name=$TPU_NAME
```

To evaluate (after predicting - prediction.json will be generated in $OUT_DIR) 
```sh
python evaluate.py $SQUAD_DIR/<test_file> <prediction_file>
```

To return a scores.json file that is used for translated SQuAD filtering, first fine-tune BERT on the handcraft dataset only (*train.json* in the *dataset* folder), then set the appropiate output folder to the location where the model is stored and run the following command.
```sh
BERT_BASE_DIR='./model/multi_cased'
SQUAD_DIR='./squad/squad_vi'
OUT_DIR='./finetuned/vi/'

!python filter_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/train_translated_squad.json \
  --do_lower_case=False \
  --output_dir=$OUT_DIR
```