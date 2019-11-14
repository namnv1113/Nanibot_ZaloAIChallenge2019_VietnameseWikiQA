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
parser.add_argument('-b', '--balance', default=True, help='The output contains the same number of data for each label')
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
    scores.loc[scores['label'] != scores['prediction'], 'probabilities'] = 1 - scores['probabilities']

    # scores = scores[scores['label'] == scores['prediction']]
    scores = scores.sort_values(by=['probabilities'], ascending=False)

    best_ids = []
    if args.balance:
        scores_false = scores[scores['label'] == 0]
        scores_true = scores[scores['label'] == 1]
        scores_false = scores_false.head(int(len(scores) * float(args.get_top_percentage) / 2))
        scores_true = scores_true.head(int(len(scores) * float(args.get_top_percentage) / 2))
        best_ids.extend(scores_false['guid'].tolist())
        best_ids.extend(scores_true['guid'].tolist())
    else:
        # Get top x% best results
        scores = scores.head(int(len(scores) * float(args.get_top_percentage)))
        best_ids.extend(scores['guid'].tolist())

    # Filter training file, store only best result
    filtered_data = list(filter(lambda item: item['id'] in best_ids, data))

    with open(args.output_file, "w", encoding=args.encoding) as json_file:
        json.dump(filtered_data, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
