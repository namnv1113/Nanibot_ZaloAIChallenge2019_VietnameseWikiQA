import json
import os
import string

with open(os.path.join(os.path.dirname(__file__), "..", "Dataset", "squad_dev_v2.0_ImpossibleAnswers.json"), 'r',
          encoding='utf-8') as infile:
    squad_json = json.load(infile)
    count_item = len(squad_json)
    count_ques = 0
    count_char = 0
    for item in squad_json:
        for para in item['paragraphs']:
            count_char = count_char + len(para['context'])
            for qes in para['qas']:
                count_ques = count_ques + 1
                count_char = count_char + len(qes)
    print(count_ques)
    print(count_char)
    infile.close()
