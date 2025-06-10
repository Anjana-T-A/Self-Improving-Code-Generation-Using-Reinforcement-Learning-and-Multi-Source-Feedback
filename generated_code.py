import re

def search_pattern(pattern, string):
    match = re.search(pattern, string)
    if match:
        return match.group(), match.start(), match.end()
    else:
        return None, None, None