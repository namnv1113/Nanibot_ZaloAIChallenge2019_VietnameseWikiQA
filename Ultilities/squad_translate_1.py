# coding=utf-8
# Run this: set GOOGLE_APPLICATION_CREDENTIALS=[PATH] on cmd"

import json
import copy
import re
from google.cloud import translate  # Imports the Google Cloud client library
import time
import html
import sys
import signal
from tqdm import tqdm
import argparse

# region Configuration
# Configuration

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input_file', default="", help='SQuAD-format file that need to be translated',
                    required=True)
parser.add_argument('-out', '--output_file', default="", help='Output file', required=True)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)
args = parser.parse_args()

error_file = 'error.txt'
progress_file = 'progress.json'
encode = args.encoding
total_case = 0
succeed_case = 0
fail_case = 0

answer_indicator = "$$$"
answer_splitter = "##"
pattern = r'\$\$\$.*?\$\$\$'

article_progress_idx = 0
paragraph_progress_idx = 0

translated_data = {'paragraphs': []}


# endregion


# region Functions
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Save current progress...')
    save()
    print("Exiting...")
    sys.exit(0)


def save():
    with open(file_output, "w", encoding=encode) as json_file:
        json.dump(translated_data, json_file, ensure_ascii=False)

    with open(progress_file, "w", encoding=encode) as json_file:
        json.dump({'article_progress': article_progress_idx, 'paragraph_progress': paragraph_progress_idx}, json_file)


def translate_func(client, text, target_lang='vi'):
    try:
        text = client.translate(text, target_language=target_lang, model='base')['translatedText']
    except KeyboardInterrupt:
        return ""
    except Exception as e:
        print('Exception: {}'.format(str(e)))
        print('Possible quota exceed. Wait for 100 + 2 seconds...')
        time.sleep(102)
        print('Sleep for 100 + 2 seconds. Continue process...')
        try:
            text = client.translate(text, target_language=target_lang, model='nmt')['translatedText']
        except:
            print('Possible daily limit exceed. Terminate the program')
            save()
            sys.exit(0)
    return html.unescape(text)


def add_info(context, ans_list):
    """ Add information to a text paragraph """
    # Split the paragragph into multiple sentences
    # Sentences structured
    # sentences = {'idx' : {start, end}, 'ans' : [{ques_id, ans_start, ans_end}], 'text' : sentence, 'org_sen_idx' : -1}
    sentences = context.split('.')
    if sentences[-1].isspace():  # If the last sentence is empty (due to the end '.')
        sentences.pop()
    sentences = [{'idx': {}, 'ans': [], 'text': sentence, 'org_sen_idx': -1} for sentence in sentences]
    for i in range(len(sentences)):
        sentences[i]['idx']['start'] = context.find(sentences[i]['text'])
        sentences[i]['idx']['end'] = sentences[i]['idx']['start'] + len(sentences[i]['text'])

    def is_same_answer_span(ans1_start_idx, ans1_end_idx, ans2_start_idx, ans2_end_idx):
        # Answers with exact same spans
        return ans1_start_idx == ans2_start_idx and ans1_end_idx == ans2_end_idx

    def is_override_answer_span(ans1_start_idx, ans1_end_idx, ans2_start_idx):
        # Answer spans override each others
        return ans1_start_idx <= ans2_start_idx <= ans1_end_idx

    # Organize information
    ans_list = sorted(ans_list, key=lambda x: (x['ans_start'], x['ans_end']), reverse=False)
    sentence_curr = 0
    for i in range(len(ans_list)):
        while sentence_curr < len(sentences) and ans_list[i]['ans_start'] > sentences[sentence_curr]['idx']['end']:
            sentence_curr += 1

        if sentence_curr >= len(sentences):
            break  # Ignore question with unknown start/end posisiton (that exceed max len)

        if len(sentences[sentence_curr]['ans']) == 0:
            sentences[sentence_curr]['ans'].append(ans_list[i])
        elif is_same_answer_span(sentences[sentence_curr]['ans'][-1]['ans_start'],
                                 sentences[sentence_curr]['ans'][-1]['ans_end'],
                                 ans_list[i]['ans_start'],
                                 ans_list[i]['ans_end']):
            sentences[sentence_curr]['ans'][-1]['ques_id'] += answer_splitter + ans_list[i]['ques_id']
        elif is_override_answer_span(sentences[sentence_curr]['ans'][-1]['ans_start'],
                                     sentences[sentence_curr]['ans'][-1]['ans_end'],
                                     ans_list[i]['ans_start']):
            # Loop over (possible) copy version of this sentence to find whether in that sentence
            # is the answer spans override each others?
            dupicate_sentence_idxs = [i for i, sentence in enumerate(sentences) if
                                      sentence['org_sen_idx'] == sentence_curr]
            alternative_found = False
            for idx in dupicate_sentence_idxs:
                sentence = sentences[idx]
                override_answers = [x for x in sentence['ans'] if
                                    is_override_answer_span(x['ans_start'], x['ans_end'], ans_list[i]['ans_start'])]
                if len(override_answers) == 0:  # No override span found
                    sentences[idx]['ans'].append(ans_list[i])
                    alternative_found = True
                    break
            # No possible duplicated sentences (if have) where span not overrided --> Create new
            if not alternative_found:
                copied_sentence = copy.deepcopy(sentences[sentence_curr])
                copied_sentence['ans'] = [ans_list[i]]
                copied_sentence['org_sen'] = sentence_curr
                sentences.insert(sentence_curr + 1, copied_sentence)
        else:
            sentences[sentence_curr]['ans'].append(ans_list[i])

    # Add information to each sentences, then append them
    new_cotext = ""
    for i in range(len(sentences)):
        sentence = sentences[i]
        sentence['ans'] = sorted(sentence['ans'], key=lambda x: (x['ans_start'], x['ans_end']), reverse=True)
        for answer in sentence['ans']:
            start_idx = answer['ans_start'] - sentence['idx']['start']
            end_idx = answer['ans_end'] - sentence['idx']['start']

            # For cased where answer: gold, Text: golden-themed
            # --> Fix answer end to be the first space after answer_end
            while end_idx < len(sentence['text']) and not sentence['text'][end_idx].isspace():
                end_idx += 1

            sentences[i]['text'] = sentences[i]['text'][:start_idx] + answer_indicator + answer['ques_id']\
                                   + answer_splitter + " " + sentences[i]['text'][start_idx:end_idx] + " " \
                                   + answer_indicator + sentences[i]['text'][end_idx:]

        sentences[i]['text'] = sentences[i]['text'].strip()
        new_cotext += sentences[i]['text'] + ". "

    return new_cotext


def load_progress():
    global article_progress_idx
    global paragraph_progress_idx
    global translated_data

    try:
        with open(progress_file, 'r', encoding=encode) as file:
            progress = json.load(file)
            article_progress_idx = progress['article_progress']
            paragraph_progress_idx = progress['paragraph_progress']
    except FileNotFoundError:
        pass

    try:
        with open(file_output, "r", encoding=encode) as json_file:
            translated_data = json.load(json_file)
    except FileNotFoundError:
        pass


# endregion


# region Main
if __name__ == "__main__":
    file_input = args.input_file
    file_output = args.output_file
    # Instantiates a client
    translate_client = translate.Client()

    # Start translating
    print('Begin translating...')
    print('Opening file...')
    with open(file_input, encoding=encode) as data_file:
        data = json.load(data_file)

    print('Load previous progress')
    load_progress()

    print('Process file and begin translate...')
    signal.signal(signal.SIGINT, signal_handler)

    for article_progress_idx in tqdm(range(article_progress_idx, len(data['data']))):
        article = data['data'][article_progress_idx]
        paragraph_progress_idx = 0
        for paragraph_progress_idx in tqdm(range(paragraph_progress_idx, len(article['paragraphs']))):
            paragraph = article['paragraphs'][paragraph_progress_idx]

            context = paragraph['context']
            answer_list = []
            questions = {}

            # Loop for each qa pairs
            for pair in paragraph['qas']:
                # Get answer and id
                answer = pair['answers'][0]['text']
                id = pair['id']
                ans_start = pair['answers'][0]['answer_start']

                # Translate the question, add to new dictionary for future use
                questions[id] = translate_func(translate_client, pair['question'])
                answer_list.append({'ques_id': id, 'ans_start': ans_start, 'ans_end': ans_start + len(answer)})

            context = add_info(context, answer_list)

            qas = []
            context = translate_func(translate_client, context)

            translated_answerlist = re.findall(pattern, context)

            # Preprocess translated paragraph and retrieve original answer
            for answer_info in translated_answerlist:

                context_idx = context.find(answer_info)  # Get answer_start position in paragraph
                _answer_info = answer_info  # Temp variable for later use
                answer_info = answer_info[
                              len(answer_indicator):-len(answer_indicator)]  # Remove the answer start and end indicator
                answer_info = answer_info.split(answer_splitter)
                real_answer = answer_info.pop().strip()  # Get answer

                for question_id in answer_info:
                    new_qdict = {'id': question_id.strip()}

                    try:
                        new_qdict['question'] = questions[new_qdict['id']]
                    except:
                        with open(error_file, "a+", encoding=encode) as err_file:
                            err_file.write(
                                'At article {}, paragarph {}: Cant find question id {}\n'.format(article_progress_idx,
                                                                                                 paragraph_progress_idx,
                                                                                                 new_qdict['id']))
                        continue

                    new_qdict['answers'] = [{'text': real_answer, 'answer_start': context_idx}]
                    qas.append(new_qdict)

                context = context.replace(_answer_info, real_answer)

            translated_data['paragraphs'].append({'context': context, 'qas': qas})

    save()
    print('Translate complete!')
# endregion
