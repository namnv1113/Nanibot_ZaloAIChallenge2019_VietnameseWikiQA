import json
import time
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from math import floor
from multiprocessing.pool import ThreadPool
import threading
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default=None,
                    help='The input English SQuAD file', required=True)
parser.add_argument('-o', '--output_file', default=None,
                    help='The output SQuAD file that have been translated into Vietnamese', required=True)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)
parser.add_argument('-t', '--num_threads', default=4,
                    help='The number of threads used for translation', required=False)


threadLocal = threading.local()
# chromepath = "/usr/bin/chromedriver"
chromepath = r"chromedriver.exe"
# chromeOptions = webdriver.ChromeOptions()
# chromeOptions.add_argument("--headless")
# driver = webdriver.Chrome(chromepath, chrome_options=chromeOptions)
# wait = WebDriverWait(driver,5)

json_result = []
count_para = 0
count_ques = 0
count_error = 0
text_error = []


def create_maindriver():
    mainDriver = getattr(threadLocal, 'maindriver', None)
    if mainDriver is None:
        chromeOptions = webdriver.ChromeOptions()
        chromeOptions.add_argument("--headless")
        mainDriver = webdriver.Chrome(chromepath, chrome_options=chromeOptions)
        # mainDriver = webdriver.Chrome(chromepath)
        setattr(threadLocal, 'maindriver', mainDriver)
    return mainDriver


def EnVieTranslationAPI(text, driver, wait):
    global count_error, text_error
    try:
        input_area = driver.find_element_by_css_selector("#source")
        input_area.clear()
        time.sleep(0.8)
        # driver.execute_script("arguments[0].value = '';", input_area)
        # driver.execute_script("arguments[0].value = '"+text+"';", input_area)
        # driver.execute_script("document.getElementById('source').setAttribute('value','"+text+"')")
        input_area.send_keys(text)
        translatedText = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                    'body > div.frame > '
                                                                    'div.page.tlid-homepage.homepage.translate-text > '
                                                                    'div.homepage-content-wrap > '
                                                                    'div.tlid-source-target.main-header > '
                                                                    'div.source-target-row > '
                                                                    'div.tlid-results-container.results-container > '
                                                                    'div.tlid-result.result-dict-wrapper > '
                                                                    'div.result.tlid-copy-target > '
                                                                    'div.text-wrap.tlid-copy-target > div > '
                                                                    'span.tlid-translation.translation'))).text
    except Exception as e:
        print("Exception: ", e)
        count_error = count_error + 1
        text_error.append(text_error)
        return ""
    return translatedText


def load_data():
    with open(args.input_file, 'r', encoding=args.encoding) as infile:
        squad_json = json.load(infile)
        squad_json = squad_json['data']

    # divide data into x parts
    div = floor(len(squad_json) / args.num_threads)
    divided_squad_json = [[squad_json[i] for i in range(div)]]
    for i in range(1, args.num_threads - 1):
        divided_squad_json.append([squad_json[j] for j in range(div * i, div * (i + 1))])
    divided_squad_json.append([squad_json[i] for i in range(div * (args.num_threads - 1), len(squad_json))])
    return divided_squad_json


def export_data(json_input):
    with open(args.output_file, 'w', encoding=args.encoding) as outfile:
        json.dump(json_input, outfile, ensure_ascii=False)
        outfile.close()


def translate_squad_vie(squad_json):
    driver = create_maindriver()
    driver.get("https://translate.google.com/?hl=vi#view=home&op=translate&sl=en&tl=vi")
    wait = WebDriverWait(driver, 20)
    print("Thread job's start!!!")
    global count_para, count_ques
    for item in squad_json:
        paragraphs = item['paragraphs']
        for para in paragraphs:
            para['context'] = EnVieTranslationAPI(para['context'], driver, wait)
            count_para = count_para + 1
            print("para: ", count_para)
            qas_list = para['qas']
            for qas in qas_list:
                count_ques = count_ques + 1
                print("ques: ", count_ques)
                qas['question'] = EnVieTranslationAPI(qas['question'], driver, wait)
    global json_result
    json_result.extend(squad_json)
    driver.quit()
    print("Thread job's done!!!")


if __name__ == "__main__":
    args = parser.parse_args()
    ThreadPool(args.num_threads).map(translate_squad_vie, load_data())
    export_data(json_result)
