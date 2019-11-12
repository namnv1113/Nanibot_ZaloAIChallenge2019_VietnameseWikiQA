import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train_file', default="", help='Zalo-format file that need to be filtered',
                    required=True)
parser.add_argument('-score', '--score_file', default="", help='Score file, produced using eval mode',
                    required=True)
parser.add_argument('-top', '--get_top_percentage', default=0.5, help='The percentage of best result to get (0 to 1)')
parser.add_argument('-output', '--output_file', default='./out.json', help='The desired, filtered output file path')
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)


def main():
    args = parser.parse_args()
    original_file = args.train_file
    score_file = args.score_file

    with open(original_file, "r", encoding=args.encoding) as data_file:
        data = json.load(data_file)

    # Filter & sort by confidence
    scores = pd.read_csv(score_file)
    scores = scores[scores['label'] == scores['prediction']]
    scores = scores.sort_values(by=['probabilities'], ascending=False)

    # Get top x% best results
    scores = scores.head(int(len(scores) * args.get_top_percentage))
    best_ids = scores['guid'].tolist()
    # Filter training file, store only best result
    filtered_data = list(filter(lambda item: item['id'] in best_ids, data))

    with open(args.output_file, "w", encoding=args.encoding) as json_file:
        json.dump(filtered_data, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
