import json
import os
import re

if __name__ == '__main__':
    """
    Helper function to check the coverage of the rules
    """
    # load anchor rules
    with open('anchor_rules.json', 'r') as f:
        anchor_rules = json.load(f)
    with open('anchor_whitelist.json', 'r') as f:
        anchor_whitelist = json.load(f)
    # load website rules
    with open('website_rules.json', 'r') as f:
        website_rules = json.load(f)
    with open('website_whitelist.json', 'r') as f:
        website_whitelist = json.load(f)
    with open('html_rules.json', 'r') as f:
        html_rules = json.load(f)

    c_total = 0
    c_filtered = 0

    # walk all websites/{doi}/one_hops directories
    for root, dirs, files in os.walk('websites'):
        for dir in dirs:
            if dir == 'one_hops':
                # read the one_hops.json file
                with open(os.path.join(root, dir, 'website.json'), 'r') as f:
                    one_hops = json.load(f)

                    c_entries = len(one_hops)
                    # filter one_hops with anchor rules
                    for rule in anchor_rules:
                        one_hops = [one_hop for one_hop in one_hops if not re.match(rule, one_hop['anchor'])]

                    # filter one_hops with website rules
                    for rule in website_rules:
                        one_hops = [one_hop for one_hop in one_hops if not re.match(rule, one_hop['website'])]

                    # filter one_hops with html rules
                    for rule in html_rules:
                        one_hops = [one_hop for one_hop in one_hops if not rule == one_hop['website']]

                    # filter one_hops with anchor whitelist
                    for rule in anchor_whitelist:
                        one_hops = [one_hop for one_hop in one_hops if not re.match(rule, one_hop['anchor'])]

                    # filter one_hops with website whitelist
                    for rule in website_whitelist:
                        one_hops = [one_hop for one_hop in one_hops if not re.match(rule, one_hop['website'])]

                    # print the rest
                    print(f'{root} \t {one_hops}')

                    c_total += c_entries
                    c_filtered += c_entries - len(one_hops)

    print(f'Number of entries: {c_total}')
    print(f'Number of filtered entries: {c_filtered}')

    # percentage of filtered entries
    print(f'Percentage of filtered entries: {c_filtered / c_total * 100}%')
