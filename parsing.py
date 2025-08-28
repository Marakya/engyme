from bs4 import BeautifulSoup
import requests
import json
import urllib3
import re
from urllib.parse import quote_plus
import os
from urllib.parse import urlparse, parse_qs
import pandas as pd
import json


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def parse_html(url):
    # url = 'https://files.stroyinf.ru/Data2/1/4293842/4293842059.htm'
    response = requests.get(url, verify=False)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    structure = {}
    current_h1 = None
    current_sub = None

    def normalize(text: str) -> str:
        """Удаляем переносы, лишние пробелы"""
        return re.sub(r'\s+', ' ', text.strip())

    for elem in soup.find_all(['h1', 'h2', 'h3','h4', 'h5', 'p']):
        if elem.name == 'h1':
            current_h1 = normalize(elem.get_text())
            structure[current_h1] = {}
            current_sub = None
        elif elem.name in ('h2', 'h3', 'h4', 'h5') and current_h1:
            current_sub = normalize(elem.get_text())
            structure[current_h1][current_sub] = ""
        elif elem.name == 'p' and current_h1:
            text = normalize(elem.get_text(" ", strip=True))
            if current_sub:
            
                if current_sub not in structure[current_h1]:
                    structure[current_h1][current_sub] = ""
                structure[current_h1][current_sub] += text + "\n"
            else:
                structure[current_h1].setdefault("Подраздел", "")
                structure[current_h1]["Подраздел"] += text + "\n"


    for h1, subs in list(structure.items()):
        for sub, text in list(subs.items()):
            if not text.strip(): 
                del structure[h1][sub]
        if not structure[h1]:      
            del structure[h1] 

    with open("data/parsed.json", "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    return structure



def image_to_json(file_path, output_path=None):
    # file_path = 'C:\\documents\\engyme\\images\\НП-068-05_Изображения.xlsx'
    # output_path = 'data_image.json'
 
    df = pd.read_excel(file_path)
    df = df.fillna('')
    data = df.to_dict(orient='records')
    # if output_path:
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=4)

    return data


def table_to_json(file_path, output_path=None):
    # file_path = 'C:\\documents\\engyme\\tables\\НП-068-05.xlsx'
    # output_path = 'data_table.json'

    sheets = pd.read_excel(file_path, sheet_name=None)
    all_data = []

    for df in sheets.values():
        df = df.fillna('') 
        data = df.to_dict(orient='records')
        all_data.extend(data)  

    return all_data



