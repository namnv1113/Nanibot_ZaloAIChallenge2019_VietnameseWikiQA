import argparse
import json
from os.path import exists
from tqdm import tqdm
from underthesea import sent_tokenize
import random

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default=None,
                    help='The input dataset file (json) with SQuAD v2.0 format', required=True)
parser.add_argument('-o', '--output_file', default="./out_zalo.json",
                    help='The desired output file (json) with Zalo format', required=False)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)
parser.add_argument('-m', '--mode', default=None, help="The conversion mode (see Readme)", required=True)
parser.add_argument('-s', '--size', default=180, required=False,
                    help="The maximum combined length of 'question' & 'text' allowed (used in mode 'short)")


def get_word_count(text):
    # Split by space & remove empty text
    texts = text.split(' ')
    try:
        text_len = len(texts.remove(""))
    except ValueError:
        text_len = len(texts)
    except TypeError:
        return 0

    return text_len


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
        for paragraph in tqdm(data['paragraphs']):
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
    for data in tqdm(squad['data']):
        for paragraph in data['paragraphs']:
            # Get paragraph split by sentences & determine its start index for easier processing
            para_context = sent_tokenize(paragraph['context'])  # Context split into list of sentences
            para_sent_startidxs = [0]   # Start index of each sentence in the paragraph
            for idx, sentence in enumerate(para_context[:-1]):
                para_sent_startidxs.append(para_sent_startidxs[idx] + len(sentence) + 1)

            # Process question-answer pairs
            for qas in paragraph['qas']:
                # Prepare data to save
                zaloQAS = {
                    'id': qas['id'],
                    'question': qas['question'],
                    'title': data['title'],
                    'label': False if qas['is_impossible'] else True
                }
                _question_len = get_word_count(qas['question'])

                # Loop & get answer text for each qa pair
                if len(qas['answers']) != 0 and qas['is_impossible'] is False \
                        and qas['answers'][0]['answer_start'] != -1:
                    # Only 1 answer, but rephrased
                    answer_start = qas['answers'][0]['answer_start']

                    # Find the sentence & sentence index that contains the answer
                    _text = None
                    _ans_sent_idx = None
                    for idx in range(len(para_context)):
                        curr_start_idx = para_sent_startidxs[idx]
                        curr_end_idx = curr_start_idx + len(para_context[idx])
                        if curr_start_idx <= answer_start <= curr_end_idx:
                            _text = para_context[idx]
                            _ans_sent_idx = idx
                            break
                        elif answer_start < curr_start_idx:
                            break
                        else:
                            continue

                    if _text is None or _ans_sent_idx is None:
                        # Problem with data --> Ignore & continue
                        print("Skip due to error")
                        continue

                    # Try to expand the answer text to reach the threshold
                    _text_len = get_word_count(_text)
                    _ans_sent_idx_before = _ans_sent_idx - 1
                    _ans_sent_idx_after = _ans_sent_idx + 1
                    while True:
                        if _ans_sent_idx_before >= 0:
                            _text_before = para_context[_ans_sent_idx_before]
                            _text_before_len = get_word_count(_text_before)
                            if _text_len + _text_before_len + _question_len <= int(args.size):
                                _text = _text_before + _text
                                _text_len += _text_before_len
                            else:
                                break
                            _ans_sent_idx_before -= 1
                        if _ans_sent_idx_after < len(para_context):
                            _text_after = para_context[_ans_sent_idx_after]
                            _text_after_len = get_word_count(_text_after)
                            if _text_len + _text_after_len + _question_len <= int(args.size):
                                _text += _text_after
                                _text_len += _text_after_len
                            else:
                                break
                            _ans_sent_idx_after += 1
                        if _ans_sent_idx_before < 0 and _ans_sent_idx_after >= len(para_context):
                            break
                    zaloQAS['text'] = _text
                else:
                    # Keep adding text until the threshold is reached
                    _text = para_context[0] if len(para_context) >= 1 else ""
                    _text_len = get_word_count(_text)
                    for idx in range(1, len(para_context)):
                        _curr_text_len = get_word_count(para_context[idx])
                        if _text_len + _question_len + _curr_text_len < int(args.size):
                            _text += para_context[idx]
                            _text_len += _curr_text_len
                        else:
                            break
                    zaloQAS['text'] = _text

                # Add data instance
                convertedData.append(zaloQAS)

    # Export converted data
    with open(output_file, 'w', encoding=encoding) as stream:
        stream.write(json.dumps(convertedData, ensure_ascii=False))


def convert_mode_veryshort(input_file, output_file, encoding):
    with open(input_file, 'r', encoding=encoding) as stream:
        squad = json.load(stream)

    convertedData = []

    # Remove _ symbol in title
    for data in squad['data']:
        data['title'] = " ".join(data['title'].split('_'))

    # Format 2: Sentence as Text
    for data in tqdm(squad['data']):
        for paragraph in data['paragraphs']:
            # Get paragraph split by sentences & determine its start index for easier processing
            para_context = sent_tokenize(paragraph['context'])  # Context split into list of sentences
            para_sent_startidxs = [paragraph['context'].index(sentence) for sentence in para_context]

            # Process question-answer pairs
            for qas in paragraph['qas']:
                # Prepare data to save
                zaloQAS = {
                    'id': qas['id'],
                    'question': qas['question'],
                    'title': data['title'],
                    'label': False if qas['is_impossible'] else True
                }
                _question_len = get_word_count(qas['question'])

                # Loop & get answer text for each qa pair
                if len(qas['answers']) != 0 and qas['is_impossible'] is False \
                        and qas['answers'][0]['answer_start'] != -1:
                    # Only 1 answer, but rephrased
                    answer = qas['answers'][0]

                    # Find the sentence & sentence index that contains the answer
                    _text = None
                    for idx in range(len(para_context)):
                        if para_sent_startidxs[idx] > answer['answer_start']:
                            continue
                        elif para_sent_startidxs[idx] < answer['answer_start'] \
                                < para_sent_startidxs[idx] + len(para_context[idx]):
                            _text = para_context[idx]
                            break
                        else:
                            break
                    zaloQAS['text'] = "" if _text is None else _text
                else:
                    zaloQAS['text'] = para_context[random.randint(0, len(para_context)) - 1] if len(para_context) >= 1 \
                        else ""

                # Add data instance
                convertedData.append(zaloQAS)

    # Export converted data
    with open(output_file, 'w', encoding=encoding) as stream:
        stream.write(json.dumps(convertedData, ensure_ascii=False))


if __name__ == "__main__":
    args = parser.parse_args()

    assert exists(args.input_file), "The input file can't be found"
    assert args.mode.lower() in ['full', 'short', 'veryshort'], "The mode can either be 'full' or 'short'"

    if args.mode.lower() == 'full':
        convert_mode_full(args.input_file, args.output_file, args.encoding)
    elif args.mode.lower() == 'short':
        convert_mode_short(args.input_file, args.output_file, args.encoding)
    elif args.mode.lower() == 'veryshort':
        convert_mode_veryshort(args.input_file, args.output_file, args.encoding)
