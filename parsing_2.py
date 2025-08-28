from bs4 import BeautifulSoup
import requests
import json
import urllib3
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = 'https://docs.secnrs.ru/catalog/FNP/NP_071_18/#'
headers = {
    'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"""
}
response = requests.get(url, headers=headers, verify=False)
html = response.text



with open("page.html", "w", encoding="utf-8") as f:
    f.write(html)

