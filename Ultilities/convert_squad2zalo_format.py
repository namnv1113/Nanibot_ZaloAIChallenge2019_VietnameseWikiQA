import argparse
import json
from os.path import exists
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default=None,
                    help='The input dataset file (json) with SQuAD v2.0 format', required=True)
parser.add_argument('-o', '--output_file', default="./out_zalo.json",
                    help='The desired output file (json) with Zalo format', required=False)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)
parser.add_argument('-m', '--mode', default=None, help="The conversion mode (see Readme)", required=True)
args = parser.parse_args()


def convert_mode_full(input_file, output_file, encoding):
    convertedData = []

    # Read data
    with open(input_file, 'r', encoding=encoding) as stream:
        squad = json.load(stream)

    # Remove _ in title
    for data in squad['data']:
        data['title'] = " ".join(data['title'].split('_'))

    # Converting
    for data in squad['data']:
        for paragraph in data['paragraphs']:
            for qas in paragraph['qas']:
                convertedData.append({
                    'id': qas['id'],
                    'question': qas['question'],
                    'title': data['title'],
                    'text': paragraph['context'],
                    'label': False if qas['is_impossible'] else True
                })

    # Export converted data
    with open(output_file, 'w', encoding=encoding) as stream:
        stream.write(json.dumps(convertedData, ensure_ascii=False))


def convert_mode_short(input_file, output_file, encoding):
    with open(input_file, 'r', encoding=encoding) as stream:
        squad = json.load(stream)

    convertedData = []

    # Remove _ symbol in title
    for data in squad['data']:
        data['title'] = " ".join(data['title'].split('_'))

    # Format 2: Sentence as Text
    for data in squad['data']:
        for paragraph in data['paragraphs']:
            for qas in paragraph['qas']:
                zaloQAS = {
                    'id': qas['id'],
                    'question': qas['question'],
                    'title': data['title'],
                    'label': False if qas['is_impossible'] else True
                }
                if len(qas['answers']) != 0:
                    for answer in qas['answers']:
                        skip = answer['answer_start']
                        foundQuestionMarkIdx = paragraph['context'][skip:].find('?')
                        foundExclamationMarkIdx = paragraph['context'][skip:].find('!')
                        foundEllipsisIdx = paragraph['context'][skip:].find('...')
                        foundDotIdx = paragraph['context'][skip:].find('.')
                        if foundExclamationMarkIdx != -1:
                            sentence = paragraph['context'][skip:].split('?')[0] + '!'
                        if foundQuestionMarkIdx != -1:
                            sentence = paragraph['context'][skip:].split('?')[0] + '?'
                        if foundEllipsisIdx != -1:
                            sentence = paragraph['context'][skip:].split('...')[0] + '...'
                        elif foundDotIdx != -1:
                            sentence = paragraph['context'][skip:].split('.')[0] + '.'
                        if sentence == '.':
                            sentence = paragraph['context'][skip + 1:].split('.')[0] + '.'
                        zaloQAS['text'] = sentence
                else:
                    cache = ".".join(paragraph['context'].split('...'))
                    cache = cache.split('.')
                    sentences = []
                    for sentence in cache:
                        if sentence != '':
                            sentences.append(sentence)
                    if sentences[-1] == '':
                        sentences.pop(-1)
                    randomSentenceIdx = randint(0, len(sentences) - 1)
                    randomSentence = sentences[randomSentenceIdx] + "."
                    if randomSentenceIdx % 2 == 0 and randomSentenceIdx < len(sentences) - 1:
                        randomSentence += " " + sentences[randomSentenceIdx + 1] + "."
                    zaloQAS['text'] = randomSentence
                convertedData.append(zaloQAS)

    # Export converted data
    with open(output_file, 'w', encoding=encoding) as stream:
        stream.write(json.dumps(convertedData, ensure_ascii=False))


if __name__ == "__main__":
    assert exists(args.input_file), "The input file can't be found"
    assert args.mode.lower() in ['full', 'short'], "The mode can either be 'full' or 'short'"

    if args.mode.lower() == 'full':
        convert_mode_full(args.input_file, args.output_file, args.encoding)
    elif args.mode.lower() == 'short':
        convert_mode_short(args.input_file, args.output_file, args.encoding)
