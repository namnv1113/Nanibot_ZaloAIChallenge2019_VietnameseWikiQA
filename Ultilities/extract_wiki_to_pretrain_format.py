import json
import argparse
from underthesea import sent_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default=None,
                    help='The input wikipedia file (each article is a json string in a line)', required=True)
parser.add_argument('-o', '--output_file', default="./pretrain_data.txt",
                    help='The desired output file (BERT pretrained data (unprocessed))', required=False)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)


if __name__ == "__main__":
    args = parser.parse_args()

    # Read file line by line & convert json string to python dict
    with open(args.input_file, 'r', encoding=args.encoding) as input_file:
        articles = input_file.readlines()

    articles = [json.loads(article, encoding=args.encoding) for article in tqdm(articles)]

    # Write to output file line by line (for each sentences), each article is split by an empty line
    with open(args.output_file, 'w', encoding=args.encoding) as output_file:
        for article in tqdm(articles):
            article_text = article["text"].replace("\r\n", ".").replace("\r", ".").replace("\n", ".")
            article_texts = sent_tokenize(article_text)
            for text in article_texts:
                output_file.write(text + "\n")
            output_file.write("\n")

    print("Completed")
