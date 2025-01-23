import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    """
    Postprocess the bounding boxes in the dataset to be multiples of 5
    """
    # walk through all  folders in /websites and load all websites.json except for the ones in the one_hops folders

    # walk through all folders in /websites
    for root, dirs, files in tqdm(os.walk('websites')):
        # for every file in this folder
        for file in files:
            # if the file is a json file and not in the one_hops folder
            if file.endswith('.json') and 'one_hops' not in root:
                # load the json
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                for i in data:
                    # "bounding_box": {"x": 0.0, "y": -37.0, "width": 1354.0, "height": 37.0}
                    bb = i['bounding_box']
                    # if bb is an array convert it to a dict
                    if type(bb) == list:
                        bb = {'x': bb[0], 'y': bb[1], 'width': bb[2], 'height': bb[3]}
                    # round the bounding box values to 5px
                    bb['x'] = round(bb['x'] / 5) * 5
                    bb['y'] = round(bb['y'] / 5) * 5
                    bb['width'] = round(bb['width'] / 5) * 5
                    bb['height'] = round(bb['height'] / 5) * 5
                    i['bounding_box'] = bb
                # save the json in the same location
                with open(os.path.join(root, file), 'w') as f:
                    json.dump(data, f)

    print('Done!')
