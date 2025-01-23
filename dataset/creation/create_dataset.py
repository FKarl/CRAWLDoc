import html
import json
import logging
import random
from tqdm import tqdm
from xml.etree import ElementTree
from lxml import etree

import os
import re
import requests

from functools import partial

from concurrent.futures import ThreadPoolExecutor


def get_puplisher_dois(percentage_of_coverage):
    """
    Get the dois of the publishers that cover a certain percentage of the total count of papers on dblp
    :param percentage_of_coverage: the percentage of coverage If the number is greater than 1, it is interpreted as the number of publishers to include.
    :return: a list of dois as strings
    """
    # load publisher.json
    with open('publisher.json') as f:
        # format: list of (doi, count on dblp, name)
        publisher_info = json.load(f)

    # sort by count (should already be sorted, but just in case)
    publisher_info.sort(key=lambda x: x[1], reverse=True)

    if percentage_of_coverage > 1:
        return [x[0] for x in publisher_info[:int(percentage_of_coverage)]]

    # get the number of dois to cover the percentage
    total_count = sum([x[1] for x in publisher_info])
    dois = []
    count = 0
    for doi, c, name in publisher_info:
        dois.append(doi)
        count += c
        if count / total_count >= percentage_of_coverage:
            break
    return dois


def process_paper_aff(paper, papers_per_publisher, dois):
    key = paper.get('key')
    # get all dois
    dois_per_paper = paper.findall('doi')
    for doi in dois_per_paper:
        # if one of the dois is in the list of dois we want to include
        doi = doi.text
        # if the doi starts with one of the dois we want to include
        if any(doi.startswith(x) for x in dois):
            # check if the doi is resolveable
            request = requests.get(f'https://doi.org/{doi}')
            status_code = request.status_code
            # 418 as iee reports that they are a teapot...
            # 403 forbidden is also allowed as acm has this status code even though the doi is valid
            if status_code not in [200, 418, 403]:
                logging.warning(f'DOI {doi} is not resolveable. Status code: {request.status_code}')
                continue

            # check publisher doi we investigate
            publisher_doi = next(x for x in dois if doi.startswith(x))

            # get the authors and their affiliations
            authors = []
            for author in paper.findall('author'):
                name = author.find('name').text
                affiliations = [aff.text for aff in author.findall('affil')]
                authors.append((name, affiliations))
            # add to papers dictionary
            papers_per_publisher[publisher_doi].append(
                {'publisher_doi': publisher_doi, 'doi': doi, 'key': key, 'authors': authors})
            # papers.append({'publisher_doi': publisher_doi, 'doi': doi, 'key': key, 'authors': authors})

            # do not add the same paper twice
            break


def get_affiliations(path_to_source, dois, samples_per_publisher):
    """
    Get the affiliations of the papers in the source
    :param path_to_source: the path to the source (xml file)
    :param dois: the dois of the publishers we want to include
    :param samples_per_publisher: the number of samples per publisher
    :return: a list of papers each containing a list of authors and their affiliations
    """

    papers_per_publisher = {doi: [] for doi in dois}

    with open(path_to_source, 'r', encoding='utf-8') as f:
        root = etree.parse(f).getroot()

        NUM_THREADS = 64
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            iter = root.findall('*')
            func = partial(process_paper_aff, papers_per_publisher=papers_per_publisher, dois=dois)
            executor.map(func, iter)

        # print number of papers per publisher
        for doi in papers_per_publisher:
            leng = len(papers_per_publisher[doi])
            logging.info(f'Publisher {doi} has {leng} papers')

        # select number of samples per publisher randomly
        missing_papers = {doi: samples_per_publisher - len(papers_per_publisher[doi]) for doi in papers_per_publisher}
        papers = []
        for doi in papers_per_publisher:
            papers_for_doi = papers_per_publisher[doi]
            if len(papers_for_doi) < samples_per_publisher:
                logging.warning(f'Not enough papers for publisher {doi}. Only {len(papers_for_doi)} found')
                papers += papers_for_doi
            else:
                papers += random.sample(papers_per_publisher[doi], samples_per_publisher)

    return papers, missing_papers


POSSIBLE_TAGS = {'article', 'inproceedings', 'book', 'phdthesis', 'incollection'}  # excluding 'proceedings'
USED_METADATA = ['title', 'year', 'publisher']


def process_paper_bib(paper, missing_data, bibliographic_data, keys, tree, missing_papers):
    key = paper.get('key')
    if key in keys:
        data = {k: (paper.find(k).text if paper.find(k) is not None else None) for k in USED_METADATA}
        # if the paper is inproceedings, get the publisher from the proceedings tag
        if paper.tag == 'inproceedings':
            # get crossref from inproceedings
            crossref = paper.find('crossref')
            if crossref is not None:
                # get the proceedings
                proceedings = tree.find(f'./proceedings[@key="{crossref.text}"]')
                if proceedings is not None:
                    # get the publisher
                    data['publisher'] = proceedings.find('publisher').text

        bibliographic_data.append({'key': key, 'data': data})
    else:
        # get publisher doi
        ee_element = paper.find('ee')
        if ee_element is not None:
            ee = ee_element.text
            # if its a doi link parse publisher doi e.g. https://doi.org/10.48550/arXiv.2211.16878
            if 'doi' in ee:
                regex_pattern = r'.*doi\.org\/(.+)\/.*'
                regex_pattern_full_doi = r'.*doi\.org\/(.+)'
                match_publisher = re.search(regex_pattern, ee)
                match_publisher_full_doi = re.search(regex_pattern_full_doi, ee)
                if match_publisher:
                    publisher_doi = match_publisher.group(1)

                    if missing_data.get(publisher_doi, 0) > 0:
                        # add the data
                        data = {k: (paper.find(k).text if paper.find(k) is not None else None) for k in
                                USED_METADATA}
                        # if the paper is inproceedings, get the publisher from the proceedings tag
                        if paper.tag == 'inproceedings':
                            # get crossref from inproceedings
                            crossref = paper.find('crossref')
                            if crossref is not None:
                                # get the proceedings
                                proceedings = tree.find(f'./proceedings[@key="{crossref.text}"]')
                                if proceedings is not None:
                                    # get the publisher
                                    data['publisher'] = proceedings.find('publisher').text

                        bibliographic_data.append({'key': key, 'data': data})

                        # also add the paper to the papers list
                        # data needed is: {'publisher_doi': '10.1109', 'doi': '10.3390/S90503337', 'key': 'conf/date/WagnerB09', 'authors': [('Ilya Wagner', ['University of Michigan, Ann Arbor, MI', 'test']), ('Valeria Bertacco', ['University of Michigan, Ann Arbor, MI'])]
                        paper_dict = {}
                        paper_dict['publisher_doi'] = publisher_doi
                        if match_publisher_full_doi:
                            paper_dict['doi'] = match_publisher_full_doi.group(1)
                        paper_dict['key'] = key
                        authors = []
                        for author in paper.findall('author'):
                            name = html.unescape(author.text)
                            # add empty affil list
                            authors.append((name, []))
                        paper_dict['authors'] = authors

                        missing_papers[publisher_doi].append(paper_dict)


def get_bibliographic_data(file_path, papers, missing_data):
    """
    Get the bibliographic data of the papers with the given keys
    :param file_path: The path to the file containing the bibliographic metadata
    :param keys: The keys of the papers we want to include
    :return: a list of papers, each containing the bibliographic metadata
    """

    keys = [paper['key'] for paper in papers]
    bibliographic_data = []

    missing_papers = {key: [] for key in missing_data}

    parser = etree.XMLParser(dtd_validation=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = etree.parse(f, parser)
        root = tree.getroot()

        NUM_THREADS = 64
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            iter = root.findall('*')
            func = partial(process_paper_bib, missing_data=missing_data, bibliographic_data=bibliographic_data,
                           keys=keys, tree=tree, missing_papers=missing_papers)
            executor.map(func, iter)

        # add the missing papers
        for publisher_doi in missing_data:
            if missing_data[publisher_doi] > 0:
                random_papers = random.sample(missing_papers[publisher_doi], missing_data[publisher_doi])
                papers += random_papers

        # merge the dicts in dataset_bib into dataset_old based on key
        for obj in papers:
            key = obj['key']
            for bib in bibliographic_data:
                if bib['key'] == key:
                    obj.update(bib)
                    break

    logging.info(missing_data)

    return bibliographic_data, papers


def parse_args():
    """
    Parse command line arguments
    :return: the parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', type=float, default=6,
                        help='The percentage of publisher coverage. If the number is greater than 1, it is interpreted as the number of publishers to include.')
    parser.add_argument('--samples_per_publisher', type=int, default=100, help='The number of samples per publisher')
    parser.add_argument('--log_level', type=str, default='INFO', help='The log level')
    parser.add_argument('--random_seed', type=int, default=1337, help='The random seed')
    parser.add_argument('--affiliation_source', type=str, default='affiliationsDblp.xml',
                        help='The source of the affiliations')
    parser.add_argument('--metadata_source', type=str, default='dblp.xml', help='The source of the metadata')
    parser.add_argument('--save_location', type=str, default='dataset.json', help='The location to save the dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    random.seed(args.random_seed)

    # get the dois of the publishers we want to include
    dois = get_puplisher_dois(args.coverage)
    logging.info(f'Create dataset with {len(dois)} publishers')

    # get the affiliations of the papers in the source
    papers, missing_papers = get_affiliations(args.affiliation_source, dois, args.samples_per_publisher)
    logging.info(missing_papers)

    # for the selected papers, get the metadata
    bibliographic_data, result = get_bibliographic_data(args.metadata_source, papers, missing_papers)

    # save the dataset
    with open(args.save_location, 'w') as f:
        json.dump(result, f)
