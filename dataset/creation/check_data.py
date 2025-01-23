import json
from tqdm import tqdm
from crossref.restful import Works

publisher = {
    "10.1109": "IEEE",
    "10.1007": "Springer",
    "10.1016": "Elsevier",
    "10.1145": "ACM",
    "10.48550": "arXiv",
    "10.3390": "MDPI"
}

if __name__ == '__main__':
    """
    Compare the data from the checked_data.json file with the data from the crossref API
    """
    # load json file
    with open('checked_data.json', 'r') as f:
        data = json.load(f)

    clean_data = []
    works = Works()

    for d in tqdm(data):
        entry = {}
        doi = d['doi']

        print("------------NEW PAPER---------------")
        print(f'Processing doi: {doi}; https://doi.org/{doi}')

        # copy simple data
        entry['doi'] = doi
        entry['publisher_doi'] = d['publisher_doi']
        entry['publisher'] = publisher[d['publisher_doi']]
        entry['year'] = d['data']['year']

        # ask crossref REST api for metadata for that api
        crossref = works.doi(doi)

        title = d['data']['title']
        # remove possible dot at the end of the title
        if title[-1] == '.':
            title = title[:-1]
        # check if the title is correct
        if crossref is not None and 'title' in crossref:
            crossref_title = crossref['title'][0]
            if title.lower() != crossref_title.lower():
                while True:
                    ans = input(f'Is the OWN title correct? OWN: "{title}", CROSSREF: "{crossref_title}" (y/n)')
                    if ans == 'n':
                        entry['title'] = crossref_title
                        break
                    elif ans == 'y':
                        entry['title'] = title
                        break
            else:
                entry['title'] = title
        else:
            entry['title'] = title

        # could be double-checked here
        entry['authors'] = d['authors']

        one_hop_websites = d['one_hop_websites']
        labeled_websites = d['labeled_websites']

        # combine the two dicts based on the website key in one_hop_websites and the first element in labeled_websites
        linked_websites = []
        for item in one_hop_websites:

            label = None
            found = False
            for k, v in labeled_websites.items():
                if item['website'] == k:
                    found = True
                    label = v
                    break
            if not found:
                # if the website was somehow not found in labeled_websites, set its label to X for manual inspection
                label = 'X'

            assert label is not None
            linked_websites.append({
                'id': item['id'],
                'anchor': item['anchor'],
                'website': item['website'],
                'label': label
            })

        entry['linked_websites'] = linked_websites

        clean_data.append(entry)

    with open('clean_data.json', 'w') as f:
        json.dump(clean_data, f, indent=4)
