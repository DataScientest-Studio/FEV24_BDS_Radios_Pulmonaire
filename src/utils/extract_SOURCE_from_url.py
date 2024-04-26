'''
extract_SOURCE_from_url.py

Définition d'une fonction pour extraire les différentes sources depuis les url fournies.
'''

def source_extract(url):
    pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
    match = pattern.search(url)
    if match:
        return match.group(1)
    else:
        return None