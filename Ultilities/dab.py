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

#passed arguments: input file path, output file path, intermediate_lang
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default=None,
                    help='The input train file', required=True)
parser.add_argument('-o', '--output_file', default=None,
                    help='The output result after back translation', required=True)
parser.add_argument('-l', '--inter_lang', default='en',
                    help='The intermediate language for back translation', required=False)
parser.add_argument('-t', '--num_threads', default=1,
                    help='The number of threads used for translation', required=False)
parser.add_argument('-e', '--encoding', default="utf-8",
                    help='The default encoding of the input/output dataset', required=False)

train_data = []
div_data = []
dab_res = []
text_error = []
count_item = 0
count_error = 0
num_item = 0


threadLocal = threading.local()
# chromepath = "/usr/bin/chromedriver"
chromepath = r"chromedriver.exe"
# chromeOptions = webdriver.ChromeOptions()
# chromeOptions.add_argument("--headless")
# driver = webdriver.Chrome(chromepath, chrome_options=chromeOptions)
# wait = WebDriverWait(driver,5)

def create_maindriver():
    mainDriver = getattr(threadLocal, 'maindriver', None)
    if mainDriver is None:
        # chromeOptions = webdriver.ChromeOptions()
        # chromeOptions.add_argument("--headless")
        # mainDriver = webdriver.Chrome(chromepath, chrome_options=chromeOptions)
        mainDriver = webdriver.Chrome(chromepath)
        setattr(threadLocal, 'maindriver', mainDriver)
    return mainDriver

def load_data():
    global train_data
    with open(args.input_file, 'r', encoding="utf-8") as infile:
        train_data = json.load(infile)
        infile.close()
    #divide input to x part
    global div_data, num_item
    num_item = len(train_data)
    div = floor(len(train_data) / args.num_threads)
    div_data.append([train_data[i] for i in range(div)])
    for i in range(1, args.num_threads - 1):
        div_data.append([train_data[j] for j in range(div*i, div*(i+1))])
    div_data.append([train_data[i] for i in range(div*(args.num_threads - 1), num_item)])    


def export_result():
    with open(args.output_file, 'w', encoding=args.encoding) as outfile:
        json.dump(dab_res, outfile, ensure_ascii=False)
        outfile.close()
def translate(text, driver, wait):
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
        text_error.append(text)
        print("Count Error: ", count_error)
        print("Text error", text_error)
        return "error"
    return translatedText

def DAB_run(data):
    print("Thread Start!!!")
    driver = create_maindriver()
    vie2inter_link = "https://translate.google.com/#view=home&op=translate&sl=vi&tl={}".format(args.inter_lang)
    inter2vie_link = "https://translate.google.com/#view=home&op=translate&sl={}&tl=vie".format(args.inter_lang)
    driver.get(vie2inter_link)
    wait = WebDriverWait(driver, 20)
    #
    global count_error, count_item
    #vie - inter
    print("---------- vie - {} ----------".format(args.inter_lang))
    for item in data:
        count_item = count_item + 1
        print("Progress: {} / {}({} x 2) = {:.4f} %".format(count_item, num_item*2, num_item, (count_item/(num_item*2))*100 ))
        item['question'] = translate(item['question'], driver, wait)
        item['text'] = translate(item['text'], driver,wait)
        
    #inter - vie
    driver.get(inter2vie_link)
    print("---------- {} - vie ----------".format(args.inter_lang))
    for item in data:
        count_item = count_item + 1
        print("Progress: {} / {}({} x 2) = {:.4f} %".format(count_item, num_item*2, num_item, (count_item/(num_item*2))*100 ))
        item['question'] = translate(item['question'], driver, wait)
        item['text'] = translate(item['text'], driver,wait)
    global dab_res
    dab_res.extend(data)
    print("Thread Done!!!")
    driver.quit()

if __name__ == "__main__":
    args = parser.parse_args()
    load_data()
    ThreadPool(args.num_threads).map(DAB_run, div_data)
    print(div_data)