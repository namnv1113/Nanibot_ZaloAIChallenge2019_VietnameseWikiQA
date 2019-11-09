
import json
import os
import codecs
import requests
import time
import selenium
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

chromepath = "/usr/bin/chromedriver"
chromeOptions = webdriver.ChromeOptions()
# chromeOptions.add_argument("--headless")
driver = webdriver.Chrome(chromepath, chrome_options=chromeOptions)
wait = WebDriverWait(driver,5)
driver.get("https://translate.google.com/?hl=vi#view=home&op=translate&sl=en&tl=vi")

with open('jquery-3.4.1.min.js','r') as jquery_js:
    jquery = jquery_js.read() #read the jquery from a file
    driver.execute_script(jquery) #active the jquery lib
    jquery_js.close()

def EnVieTranslationAPI(text):
    input_area = driver.find_element_by_css_selector("#source")
    # input_area.clear()
    driver.execute_script("document.getElementById('source').value = '';")
    text = text.replace("'", "\'")
    driver.execute_script("$('#source').val('"+text+"')")
    # input_area.send_keys(text)
    translatedText =  wait.until(EC.presence_of_element_located((By.CSS_SELECTOR , 'body > div.frame > div.page.tlid-homepage.homepage.translate-text > div.homepage-content-wrap > div.tlid-source-target.main-header > div.source-target-row > div.tlid-results-container.results-container > div.tlid-result.result-dict-wrapper > div.result.tlid-copy-target > div.text-wrap.tlid-copy-target > div > span.tlid-translation.translation'))).text    
    return translatedText
    
def translate_squad_vie():
    with open(os.path.join(os.path.dirname(__file__), "..","Dataset", "squad_dev_v2.0_ImpossibleAnswers.json"),'r') as infile:
        squad_json = json.load(infile)
        infile.close()
    count_para = 0
    count_ques = 0
    with  open(os.path.join(os.path.dirname(__file__), "..","Dataset", "vie_squad_dev_v2.0_ImpossibleAnswers.json"),'w') as outfile:
        for item in squad_json:
            paragraphs = item['paragraphs']
            for para in paragraphs:
                para['context'] = EnVieTranslationAPI(para['context'])
                count_para = count_para + 1
                print("para: ", count_para)
                qas_list = para['qas']
                for qas in qas_list:
                    count_ques = count_ques + 1
                    print("ques: ", count_ques)
                    qas['question'] = EnVieTranslationAPI(qas['question'])
        json.dump(squad_json, outfile)
        outfile.close()
if __name__ == "__main__":
    translate_squad_vie()
    driver.quit()
