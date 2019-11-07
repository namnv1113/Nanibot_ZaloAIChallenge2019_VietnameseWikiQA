import pandas as pd
import json
from os import getcwd



if __name__ == '__main__':
    convertedData = []

    # Get file path & read data
    filePath = str(input("Input data file path: "))
    with open(filePath, 'r') as stream:
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
    with open(getcwd() + '\\Dataset\\SQUAD_Format1.json', 'w') as stream:
        stream.write(json.dumps(convertedData))