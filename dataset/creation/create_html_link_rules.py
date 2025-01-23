import os
from bs4 import BeautifulSoup
import json

from urllib.parse import urlparse

if __name__ == '__main__':
    blacklisted_links = []
    # rules per publisher
    css_path_rules = {
        '10.1007': [
            'div#chapter-info-section.c-article-section div#chapter-info-content.c-article-section__content',
            'div.app-elements.u-mb-16 header.eds-c-header',
            'article div.c-article-body div.c-article-buy-box--article.u-mb-24 div#sprcom-buybox.sprcom-buybox.buybox div div.c-box',
            'div#sidebar.c-article-extras.u-text-sm.u-hide-print',
            'article div.c-article-body div#MagazineFulltextChapterBodySuffix section div#Bib1-section.c-article-section div#Bib1-content.c-article-section__content',
            'div.c-article-body div.main-content',
            'section.c-article-recommendations',
            'section div#Bib1-section.c-article-section',
            'main.c-article-main-column.u-float-left.js-main-column.u-text-sans-serif section',
            'html.js body.shared-article-renderer div.u-vh-full div#main-content.u-container.u-mb-32.u-clearfix main.c-article-main-column.u-float-left.js-main-column.u-text-sans-serif section div.c-box.c-box--shadowed',
            'article div.c-article-body div#MagazineFulltextChapterBodySuffix'

        ],
        '10.1016': [
            'div#section-cited-by',
            'section.bibliography',
            'section.RelatedContentPanel',
            'footer',
            'html body.toolbar-stuck div div#app.App div.page div.sd-flex-container div.sd-flex-content header#gh-cnt',
            'html body div div#app.App div.page div.sd-flex-container div.sd-flex-content div#mathjax-container.Article div.article-wrapper.grid.row article.col-lg-12.col-md-16.pad-left.pad-right.u-padding-s-top div#body.Body'
        ],
        '10.1109': [
            'header',
            'section#xploreFooter',
            'footer xpl-footer div#xplore-footer.stats-footer',
            'div.document-sidebar-rel-art',
            'section div.document-full-text-content',
            'section#references-anchor.document-all-references'

        ],
        '10.1145': [
            'div header.header',
            'div.article__section.article__references',
            'div.article__body.article__abstractView div.pb-dropzone',
            'div.citation.article__section.article__index-terms',
            'div.recommended-articles__content',
            'div#pill-references',
        ],
        '10.3390': [
            'section.main-section header',
            'div.middle-column__help__fixed.show-for-medium-up.affix-top',
            'div#left-column.content__column',
            'div.html-article-content div.hypothesis_container div.html-body',
            'section#html-references_list',
            'div#footer'

        ],
        '10.48550': [
            'div#labstabs div.labstabs',
            'header',
            'div.submission-history'
        ],
    }

    base_urls = {
        '10.1007': 'https://link.springer.com',
        '10.1016': 'https://www.sciencedirect.com',
        '10.1109': 'https://ieeexplore.ieee.org',
        '10.1145': 'https://dl.acm.org',
        '10.3390': 'https://www.mdpi.com',
        '10.48550': 'https://www.arxiv.org'
    }

    # walk through all websites/{doi}/website.html files but not the subfolders
    for root, dirs, files in os.walk('websites'):
        for file in files:
            # if root ends wiht site_{number}, then skip
            if 'site_' in root:
                continue
            if file == 'website.html':
                print(root)
                doi = root.split('\\')[1]  # for linux, use '/' and for windows use '\\'
                print(f'Processing {doi}')
                # with file in utf-8 mode
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    website = f.read()
                    soup = BeautifulSoup(website, 'html.parser')
                    # for each publisher, check if the website contains any blacklisted links
                    for rule in css_path_rules[doi]:
                        elements = soup.select(rule)
                        for element in elements:
                            # check if there is a subelement that is a link
                            links = element.find_all('a', href=True)
                            for link in links:
                                url = link['href']

                                # convert relative link to absolute link
                                if not url.startswith('http'):
                                    base_url = base_urls[doi]
                                    url = base_url + url

                                blacklisted_links.append(url)

    # make blacklisted links unique
    blacklisted_links = list(set(blacklisted_links))
    # save as json
    with open('html_rules.json', 'w') as f:
        json.dump(blacklisted_links, f)

    print('Number of blacklisted links:', len(blacklisted_links))
