import csv
import sys
import pandas as pd
import pdb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []

    months_dict = { 'Jan':0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6,
                   'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    csv_data = pd.read_csv(filename)
    for index, row in csv_data.iterrows():
        values = {
            'Administrative': int(row['Administrative']),
            'Informational': int(row['Informational']),
            'ProductRelated': int(row['ProductRelated']),
            'OperatingSystems': int(row['OperatingSystems']),
            'Month': row['Month'],
            'Browser': int(row['Browser']),
            'Region': int(row['Region']),
            'TrafficType': int(row['TrafficType']),
            'VisitorType': row['VisitorType'],
            'Weekend': row['Weekend'],
            'Administrative_Duration': float(row['Administrative_Duration']),
            'Informational_Duration': float(row['Informational_Duration']),
            'ProductRelated_Duration': float(row['ProductRelated_Duration']),
            'BounceRates': float(row['BounceRates']),
            'ExitRates': float(row['ExitRates']),
            'PageValues': float(row['PageValues']),
            'SpecialDay': float(row['SpecialDay'])

        }

        revenue = row['Revenue']
        labels.append(1) if revenue == True else labels.append(0)

        for type, val in values.items():
            if type == 'Month':
                values[type] = months_dict[val]

            elif type == 'Weekend':
                values[type] = 1 if val == True else 0

            elif type == 'VisitorType':
                values[type] = 1 if val == 'Returning_Visitor' else 0

        columns = csv_data.columns
        row_of_evidence = []
        for col in columns:
            if col == 'Revenue':
                break

            row_of_evidence.append(values[col])
        
        evidence.append(row_of_evidence)

    return (evidence, labels)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    outcome = model.fit(evidence, labels)
    return outcome



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # positive revenue accurately identified
    correct_positive = 0
    actual_positive_total = 0

    # negative revenue accurately identified
    correct_negative = 0
    actual_negative_total = 0

    for revenue, hypothesis in zip(labels, predictions):
        if revenue == 0 and revenue == hypothesis:
            correct_positive += 1
            actual_positive_total += 1
        elif revenue == 0:
            actual_positive_total += 1
        
        if revenue == 1 and revenue == hypothesis:
            correct_negative += 1
            actual_negative_total += 1
        elif revenue == 1:
            actual_negative_total += 1

    sensitivity = float(correct_positive/actual_positive_total)
    specificity = float(correct_negative/actual_negative_total)
    return (sensitivity, specificity)
    


if __name__ == "__main__":
    main()
