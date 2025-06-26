import re
from datetime import datetime

def extract_date(url):
    regex = r"(\d{4})/(\d{2})/(\d{2})/"
    result = re.search(regex, url)
    year = int(result.group(1))
    month = int(result.group(2))
    date = int(result.group(3))
    return {'year': year, 'month': month, 'date': date}