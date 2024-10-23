import requests
from pprint import pprint
from bs4 import BeautifulSoup

URL = "https://web.archive.org/web/20220928064431/http://www.rcti.cteguj.in/facultydetail/show/37220"

resp = requests.get(URL)

soup = BeautifulSoup(resp.content.decode('utf-8'), "html.parser")
print(soup.prettify())

# CHA MODAI GAI!!!!!!!!!!!!