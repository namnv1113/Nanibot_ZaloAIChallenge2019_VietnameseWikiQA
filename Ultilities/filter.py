import json
import io
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train_file', default="", help='SQuAD-format files that need to be filtered',
                    required=True)
parser.add_argument('-score', '--score_file', default="", help='Score file, produced using squad_filter.py',
                    required=True)
args = parser.parse_args()


def main():
    original_file = args.train_file
    score_file = args.score_file

    with open(original_file, "r", encoding='utf-8-sig') as data_file:
        data = json.load(data_file)

    with open(score_file, "r", encoding='utf-8') as data_file:
        scores = json.load(data_file)

    ids = []
    # ids = [value for (key, value) in sorted(scores.items())]
    for w in sorted(scores, key=scores.get, reverse=True):
        ids.append(w)
    ids_25 = ids[0: int(len(ids) * 0.25)]
    ids_50 = ids[0: int(len(ids) * 0.5)]
    ids_75 = ids[0: int(len(ids) * 0.75)]

    data_25 = {'data': [{'title': "data25", 'paragraphs': []}]}
    data_50 = {'data': [{'title': "data50", 'paragraphs': []}]}
    data_75 = {'data': [{'title': "data75", 'paragraphs': []}]}

    for paragraphs in tqdm(data['data']):
        for paragraph in tqdm(paragraphs['paragraphs']):
            qas_25 = []
            qas_50 = []
            qas_75 = []
            for question in paragraph['qas']:
                if question['id'] in ids_25:
                    qas_25.append(question)
                if (question['id'] in ids_50):
                    qas_50.append(question)
                if (question['id'] in ids_75):
                    qas_75.append(question)
            data_25['data'][0]['paragraphs'].append({'context': paragraph['context'], 'qas': qas_25})
            data_50['data'][0]['paragraphs'].append({'context': paragraph['context'], 'qas': qas_50})
            data_75['data'][0]['paragraphs'].append({'context': paragraph['context'], 'qas': qas_75})

    with io.open('result_25.json', "w", encoding='utf-8') as json_file:
        json.dump(data_25, json_file, ensure_ascii=False)

    with io.open('result_50.json', "w", encoding='utf-8') as json_file:
        json.dump(data_50, json_file, ensure_ascii=False)

    with io.open('result_75.json', "w", encoding='utf-8') as json_file:
        json.dump(data_75, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
