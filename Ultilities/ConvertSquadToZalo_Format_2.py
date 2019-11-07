import pandas as pd
import json
from os import getcwd
from random import randint


if __name__ == '__main__':
    filePath = str(input("Input data file path: "))
    with open(filePath, 'r') as stream:
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
                        if (foundExclamationMarkIdx != -1):
                            sentence = paragraph['context'][skip:].split('?')[0] + '!'
                        if (foundQuestionMarkIdx != -1):
                            sentence = paragraph['context'][skip:].split('?')[0] + '?'
                        if (foundEllipsisIdx != -1):
                            sentence = paragraph['context'][skip:].split('...')[0] + '...'
                        elif (foundDotIdx != -1):
                            sentence = paragraph['context'][skip:].split('.')[0] + '.'
                        if (sentence == '.'):
                            sentence = paragraph['context'][skip+1:].split('.')[0] + '.'
                        zaloQAS['text'] = sentence
                else:
                    cache = ".".join(paragraph['context'].split('...'))
                    cache = cache.split('.')
                    sentences = []
                    for sentence in cache:
                        if (sentence != ''):
                            sentences.append(sentence)
                    if (sentences[-1] == ''):
                        sentences.pop(-1)
                    randomSentenceIdx = randint(0, len(sentences)-1)
                    randomSentence = sentences[randomSentenceIdx] + "."
                    if randomSentenceIdx % 2 == 0 and randomSentenceIdx < len(sentences)-1:
                        randomSentence += " " + sentences[randomSentenceIdx + 1] + "."
                    zaloQAS['text'] = randomSentence
                convertedData.append(zaloQAS)

    # Export converted data
    with open(getcwd() + '\\Dataset\\SQUAD_Format2.json', 'w') as stream:
        stream.write(json.dumps(convertedData))
