import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    """
    Create plots and statistics for the dataset
    """
    with open('clean_data.json', 'r') as f:
        data = json.load(f)

    year_count = {}
    author_count = {}
    number_of_affiliations = {}
    num_linked_websites = {}
    total_linked_websites = 0
    num_of_impotant_websites = {}

    for d in tqdm(data):
        number_of_authors = len(d['authors'])
        year = d['year']

        if year in year_count:
            year_count[year] += 1
        else:
            year_count[year] = 1

        if number_of_authors in author_count:
            author_count[number_of_authors] += 1
        else:
            author_count[number_of_authors] = 1

        for author in d['authors']:
            num_aff = len(author[1])
            if num_aff in number_of_affiliations:
                number_of_affiliations[num_aff] += 1
            else:
                number_of_affiliations[num_aff] = 1

        num_lw = len(d['linked_websites'])
        total_linked_websites += num_lw
        if num_lw in num_linked_websites:
            num_linked_websites[num_lw] += 1
        else:
            num_linked_websites[num_lw] = 1

        c = 0
        for website in d['linked_websites']:
            if website['label'] == 1:
                c += 1

        if c in num_of_impotant_websites:
            num_of_impotant_websites[c] += 1
        else:
            num_of_impotant_websites[c] = 1

    # dict to representative list
    author_list = []
    for k, v in author_count.items():
        author_list += [k] * v
    affiliation_list = []
    for k, v in number_of_affiliations.items():
        affiliation_list += [k] * v
    linked_websites_list = []
    for k, v in num_linked_websites.items():
        linked_websites_list += [k] * v
    important_websites_list = []
    for k, v in num_of_impotant_websites.items():
        important_websites_list += [k] * v

    print("--------TOTALS-----------")
    print("Total Number of publications: ", len(data))
    print("Total number of authors: ", sum([k * v for k, v in author_count.items()]))
    print("Total number of affiliations: ", sum([k * v for k, v in number_of_affiliations.items()]))
    print("Total number of linked websites: ", total_linked_websites)
    # print averages
    print("--------AVERAGES-----------")
    print("Average number of authors: ", sum([k * v for k, v in author_count.items()]) / sum(author_count.values()),
          "Standard deviation: ", np.std(author_list))
    print("Average number of affiliations per author: ",
          sum([k * v for k, v in number_of_affiliations.items()]) / sum(number_of_affiliations.values()),
          "Standard deviation: ", np.std(affiliation_list))
    print("Average number of linked websites: ",
          sum([k * v for k, v in num_linked_websites.items()]) / sum(num_linked_websites.values()),
          "Standard deviation: ", np.std(linked_websites_list))
    print("Average number of important websites: ",
          sum([k * v for k, v in num_of_impotant_websites.items()]) / sum(num_of_impotant_websites.values()),
          "Standard deviation: ", np.std(important_websites_list))

    print("--------RAW DATA-----------")
    print("Number of publications per year: ", year_count)
    print("Number of authors: ", author_count)
    print("Number of affiliations: ", number_of_affiliations)
    print("Number of linked websites: ", num_linked_websites)
    print("Number of important websites: ", num_of_impotant_websites)

    # plot distributions

    # Convert the keys to integers
    year_count = {int(year): count for year, count in year_count.items()}

    # Sort the keys and values in year_count
    years = sorted(year_count.keys())
    counts = [year_count[year] for year in years]

    # Plot the data
    plt.bar(years, counts)
    plt.xlabel('Year')
    plt.ylabel('Number of publications')
    # plt.title('Number of publications per year')
    plt.show()

    plt.bar(author_count.keys(), author_count.values())
    plt.xlabel('Number of authors')
    plt.ylabel('Number of publications')
    # plt.title('Number of authors per paper')
    plt.show()

    plt.bar(number_of_affiliations.keys(), number_of_affiliations.values())
    plt.xlabel('Number of affiliations')
    plt.ylabel('Number of authors')
    # plt.title('Number of affiliations per author')
    plt.show()

    plt.bar(num_linked_websites.keys(), num_linked_websites.values())
    plt.xlabel('Number of linked websites')
    plt.ylabel('Number of publications')
    # plt.title('Number of linked websites per paper')
    plt.show()

    # Convert the keys to integers and get the values
    keys = [int(k) for k in num_linked_websites.keys()]
    values = list(num_linked_websites.values())

    # Define the number of bins you want
    num_bins = 25

    # Plot the histogram
    plt.hist(keys, bins=num_bins, weights=values, edgecolor='black')
    plt.xlabel('Number of linked websites')
    plt.ylabel('Number of publications')
    # plt.title('Number of linked websites per paper')
    plt.show()

    plt.bar(num_of_impotant_websites.keys(), num_of_impotant_websites.values())
    plt.xlabel('Number of important websites')
    plt.ylabel('Number of publications')
    # plt.title('Number of important websites per paper')
    plt.show()
