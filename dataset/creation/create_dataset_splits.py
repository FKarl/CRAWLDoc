import json
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Load the data
    with open('clean_data.json', 'r') as f:
        data = json.load(f)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Initialize empty lists for the train, validation, and test sets
    train = []
    validation = []
    test = []

    # Group the DataFrame by the publisher
    for _, group in df.groupby('publisher_doi'):
        # Split the group into train, validation, and test sets
        train_group, test_group = train_test_split(group, test_size=0.2, random_state=42)
        validation_group, test_group = train_test_split(test_group, test_size=0.5, random_state=42)

        # Append the group splits to the corresponding sets
        train.append(train_group)
        validation.append(validation_group)
        test.append(test_group)

    # Concatenate the splits from all groups
    train = pd.concat(train)
    validation = pd.concat(validation)
    test = pd.concat(test)

    # Check dimensions
    print(f"Train set: {len(train)}")
    print(f"Validation set: {len(validation)}")
    print(f"Test set: {len(test)}")

    # Save the split data
    train.to_json('train.json', orient='records')
    validation.to_json('validation.json', orient='records')
    test.to_json('test.json', orient='records')

    # Get unique publishers
    publishers = df['publisher_doi'].unique()

    # Create a split for each publisher
    for publisher in publishers:
        # Split the data
        test = df[df['publisher_doi'] == publisher]
        train = df[df['publisher_doi'] != publisher]

        # Save the split data
        train.to_json(f'train_all_except_{publisher}.json', orient='records')
        test.to_json(f'test_{publisher}.json', orient='records')
