import json
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def calculate_accuracy_and_f1_score(predicted_file, true_labels_file):
    # Read predicted labels from the TSV file
    predicted_labels = {}
    with open(predicted_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            predicted_labels[parts[0]] = parts[1]

    # Read true labels from the JSON file
    with open(true_labels_file, 'r') as f:
        true_labels = json.load(f)

    # Convert labels to lists
    true_labels_list = []
    predicted_labels_list = []
    for idx, row in enumerate(true_labels):
        name = row['id']
        true_label = row['class_label']
        predicted_label = predicted_labels.get(name)
        if predicted_label is not None:  # Ensure prediction exists
            true_labels_list.append(true_label)
            predicted_labels_list.append(predicted_label)

    # Calculate accuracy
    accuracy = sum(1 for true_label, predicted_label in zip(true_labels_list, predicted_labels_list) if true_label == predicted_label) / len(true_labels_list)

    # Calculate F1-score
    f1 = f1_score(true_labels_list, predicted_labels_list, average='macro')
    precision = precision_score(true_labels_list, predicted_labels_list, average='weighted')
    recall = recall_score(true_labels_list, predicted_labels_list, average='weighted')
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)
    print(conf_matrix)
    # Extract values from the confusion matrix
    incorrect_P = conf_matrix[1][0]  # Propaganda misclassified as non-propaganda
    incorrect_NP = conf_matrix[0][1]  # Non-propaganda misclassified as propaganda

    # Print the number of incorrectly classified P and NP samples
    print(f'Number of incorrectly classified propaganda samples: {incorrect_P}')
    print(f'Number of incorrectly classified non-propaganda samples: {incorrect_NP}')

    return accuracy, f1, precision, recall

# Example usage
predicted_file = '/home/slasher/araieval_arabicnlp24/task2/baselines/task2C_MODOS.tsv'
true_labels_file = '/home/slasher/araieval_arabicnlp24/task2/data/arabic_memes_propaganda_araieval_24_test_gold.json'

accuracy, f1,p,r = calculate_accuracy_and_f1_score(predicted_file, true_labels_file)
print("Accuracy:", accuracy)
print("Macro F1-score:", f1)
print("  precesion :", p)
print(" recall: ", r)
