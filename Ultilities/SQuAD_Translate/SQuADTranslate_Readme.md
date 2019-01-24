# SQuAD Translate

This folder contains the source code to translate SQuAD training file (from English to Vietnamese)

## Requirements
	* Python 3.x (Tested on Python 3.6.7)
	* copy
	* html
	* tqdm
	* google.cloud (Authentication is required, which is not included in this CD)

## Run the code

```sh
export GOOGLE_APPLICATION_CREDENTIALS="<path_to_authenticate_file>"
python translate.py -in <input_file> -out <output_file>
```

Along with the output translated file, *error.txt* and *progress.json* are returned indicates question-answer pairs with errors that can be processes; and current progress so that the program can be continued after termination.