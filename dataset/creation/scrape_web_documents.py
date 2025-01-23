from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import os
import json
import sys
import shutil

import html
from urllib.parse import urlparse

import requests
import time

from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser

from concurrent.futures import ThreadPoolExecutor


def resolve_doi(doi):
    """
    Resolves the doi to the url
    :param doi: the doi
    :return: the url
    """
    link = f'https://doi.org/{doi}'
    website_url = requests.get(link).url
    return website_url


def extract_text_and_bounding_boxes_from_website(driver):
    """
    Extracts the text and the bounding boxes of the text from the website
    :param url: the url of the website
    :return: dictionary containing the text and the bounding boxes
    """

    print("Extracting text and bounding boxes from website")

    data = []

    elements = driver.find_elements(By.CSS_SELECTOR, '*')  # get all elements
    for element in elements:
        try:
            # Check if the element has any child elements
            if not element.find_elements(By.XPATH, './/*'):
                text = element.text
                bounding_box = element.rect  # get bounding box (x, y, width, height)
                # Check if the text is not empty before printing
                if text.strip():
                    data.append({'text': text, 'bounding_box': bounding_box})
        except Exception as e:
            continue

    print("Finished extraction")

    return data


def extract_text_with_layout_from_pdf(url, pdf_path, driver):
    """
    Extracts the text and the bounding boxes of the text from the pdf
    """

    print("Extracting text and bounding boxes from pdf")

    data = []

    # create pdf path if it does not exist
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # get the pdf file
    with open(pdf_path, 'wb') as file:
        # copy cookies from the selenium driver to the requests session
        cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
        response = requests.get(url, cookies=cookies)
        if response.status_code not in [200, 201, 202]:
            print(f"Could not download pdf. Status code: {response.status_code}")
            return []
        file.write(response.content)

    print("Downloaded pdf")

    with open(pdf_path, 'rb') as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(resource_manager, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, device)

        print("Starting extraction")

        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            for element in layout:
                if isinstance(element, LTTextBox) or isinstance(element, LTTextLine):
                    data.append({'text': element.get_text(), 'bounding_box': element.bbox})

    print("Finished extraction")

    return data


def get_onehop(driver):
    """
    Get the one-hop website links of the current website
    :param driver: the selenium web driver
    :return: a list of all outgoing links
    """
    try:
        # get all links
        links = driver.find_elements(By.TAG_NAME, 'a')
        hrefs = []

        for link in links:
            try:
                href = link.get_attribute('href')

                # check if the link is outgoing
                if href and href.startswith('http'):
                    anchor_text = link.text
                    anchor_text = html.unescape(anchor_text)

                    hrefs.append((anchor_text, href))
            except Exception as e:
                continue

        # remove duplicate links even if they have different anchor text
        # also remove links that are same without #...
        link_set = set()
        final_links = []
        for text, url in hrefs:
            # Parse the URL and rebuild it without the fragment
            parsed_url = urlparse(url)
            url_without_fragment = parsed_url._replace(fragment="").geturl()

            if url_without_fragment not in link_set:
                link_set.add(url_without_fragment)
                final_links.append((text, url_without_fragment))

        return final_links
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def save_to_json(data, path):
    """
    Save the data to a json file
    :param data: the data to be saved
    :param path: the path to save the data
    """
    # create path
    os.makedirs(path, exist_ok=True)

    with open(f'{path}/website.json', 'w') as file:
        json.dump(data, file)


def remove_cookie_banner(driver):
    already_accepted = False
    # accept cookies if possible

    # springer
    if not already_accepted:
        try:
            driver.find_element(By.CLASS_NAME, 'cc-banner__button-accept').click()
            already_accepted = True
        except:
            pass

    # mdpi and acm
    if not already_accepted:
        try:
            driver.find_element(By.ID, 'CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll').click()
            already_accepted = True
        except:
            pass

    # ieee (and wiley)
    if not already_accepted:
        try:
            element = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'osano-cm-button--type_accept'))
            )
            element.click()
            already_accepted = True
        except:
            pass

    # elsevier
    if not already_accepted:
        try:
            element = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))
            )
            element.click()
            already_accepted = True

            # remove the onetrust-pc-dark-filter that still persists after closing the banner <div class="onetrust-pc-dark-filter ot-fade-in">
            driver.execute_script("document.querySelector('.onetrust-pc-dark-filter').remove();")
        except:
            pass

    # ps arxiv does not have a cookie banner


def open_parts(driver, paper):
    # ieee
    if paper['publisher_doi'] == '10.1109':
        try:
            element = WebDriverWait(driver, 1).until(
                # link that contains the text "All Authors"
                EC.element_to_be_clickable((By.XPATH, '//*[contains(text(), "All Authors")]'))
            )
            element.click()
        except:
            pass
    # springer does not have parts to open
    # elsevier
    elif paper['publisher_doi'] == '10.1016':
        # click show more with id show-more-btn
        try:
            element = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.XPATH, '//*[contains(@id, "show-more-btn")]'))
            )
            element.click()
        except Exception as e:
            pass


def process_paper(paper):
    try:

        try:
            options = FirefoxOptions()

            options.set_preference("browser.download.folderList", 2)
            options.set_preference("browser.download.manager.showWhenStarting", False)
            options.set_preference("browser.download.dir", os.path.abspath('tmp'))
            options.set_preference("browser.download.alwaysOpenPanel", False)
            options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
            options.set_preference("browser.link.open_newwindow", 3)
            # options.add_argument("--headless")

            driver = webdriver.Firefox(options=options)
        except Exception as e:
            print(f"An error occurred: {e}")

            # try to close the driver
            try:
                driver.quit()
            except:
                pass

            return

        landing_page = resolve_doi(paper['doi'])  # open the landing_page
        try:
            driver.get(landing_page)
        except TimeoutException:
            print(f"Loading {website} timed out. Skipping to next website.")
            return

        # remove cookie banner
        remove_cookie_banner(driver)

        # open parts of the website
        open_parts(driver, paper)

        # now for each one_hop_website
        # check if it already exists
        if not os.path.exists(f"websites/{paper['doi']}/one_hops/website.json"):
            # get all one_hop_websites
            one_hop_websites = get_onehop(driver)

            counter = 0
            one_hops = []
            for anchor, website in one_hop_websites:
                one_hops.append({'anchor': anchor, 'website': website, 'id': counter})
                counter += 1
            save_to_json(one_hops, f"websites/{paper['doi']}/one_hops")
        else:
            print(f"one_hops for {paper['doi']} already exists. Skipping.")
            one_hops = json.load(open(f"websites/{paper['doi']}/one_hops/website.json", 'r'))

        # save landing page in folder with url name
        if not os.path.exists(f"websites/{paper['doi']}/website.json"):
            landing_page_informations = extract_text_and_bounding_boxes_from_website(driver)
            save_to_json(landing_page_informations, f"websites/{paper['doi']}")
        # also save html
        if not os.path.exists(f"websites/{paper['doi']}/website.html"):
            with open(f"websites/{paper['doi']}/website.html", 'w', encoding='utf-8') as file:
                page_source = str(driver.page_source)
                file.write(page_source)

        for iter_website in one_hops:
            anchor = iter_website['anchor']
            website = iter_website['website']
            counter = iter_website['id']

            pdf_path = f"websites/{paper['doi']}/site_{counter}/pdf.pdf"

            # check if the website is already processed / folder already exists
            if os.path.exists(f"websites/{paper['doi']}/site_{counter}"):
                print(f"site_{counter} for {paper['doi']} already exists. Skipping.")
                continue

            try:
                # if the website is a pdf ending or has /pdf in the url
                if website.endswith('.pdf') or website.endswith('.PDF') or website.endswith(
                        '.Pdf') or '/pdf' in website or anchor.lower() == 'pdf':
                    website_information = extract_text_with_layout_from_pdf(website, pdf_path, driver)
                else:
                    # it's a website
                    try:
                        driver.get(website)
                    except TimeoutException:
                        print(f"Loading {website} timed out. Skipping to next website.")
                        continue

                    # remove cookie banner
                    remove_cookie_banner(driver)

                    # open parts of the website
                    open_parts(driver, paper)

                    # also save html
                    os.makedirs(f"websites/{paper['doi']}/site_{counter}", exist_ok=True)
                    with open(f"websites/{paper['doi']}/site_{counter}/website.html", 'w', encoding='utf-8') as file:
                        page_source = str(driver.page_source)
                        file.write(page_source)

                    website_information = extract_text_and_bounding_boxes_from_website(driver)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            save_to_json(website_information, f"websites/{paper['doi']}/site_{counter}")

        print(f"Finished processing {paper['doi']}")
        driver.quit()


    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        # try to close the driver
        try:
            driver.quit()
        except:
            pass

        return


if __name__ == '__main__':
    """
    Scrape all Web documents (HTML and PDF) for the dataset
    """
    os.makedirs('tmp', exist_ok=True)

    # load  dataset.json
    with open('dataset.json', 'r') as file:
        data = json.load(file)

    NUM_THREADS = 64

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(process_paper, data)

    print("Finished processing all papers")
