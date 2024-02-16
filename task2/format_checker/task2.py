import os
import argparse
import logging
from pathlib import Path

"""
This script checks whether the results format for Task 2 (2A and 2B) is correct. 
It also provides some warnings about possible errors.

The submission of the result file for subtask 2A should be in TSV format. 
One row of the prediction file is the following:
id <TAB> label

where id is the tweet id as given in the test file, and label is the predicted binary label, either "disinfo" or "no-disinfo".
For example:

22	disinfo
23	no-disinfo
24	no-disinfo

=====================

The submission of the result file for subtask 2B should be in TSV format. 
One row of the prediction file is the following:
id <TAB> label

where id is the tweet id as given in the test file, and label is the predicted class, either "Rumor", "HS", "OFF" or "SPAM". 
For example:

22	HS
23	Rumor
24	OFF
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def check_format(file_path, subtask):
    logging.info("Checking format of prediction file...")

    if not os.path.exists(file_path):
        logging.error("File doesn't exist: {}".format(file_path))
        return False

    with open(file_path, encoding='UTF-8') as out:
        next(out)
        file_content = out.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            doc_id, labels = line.strip().split('\t')

            if subtask == "2A":
                if labels not in ["disinfo", "no-disinfo"]:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False

            if subtask == "2B":
                if labels not in ["OFF", "Rumor", "SPAM", "HS"]:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False

    logging.info("File passed format checker!")

    return True


def validate_files(pred_files_path):
    logging.info("Validating if passed files exist...")

    for pred_file_path in pred_files_path:
        if not os.path.exists(pred_file_path):
            logging.error("File doesn't exist: {}".format(pred_file_path))
            return False

        # Check if the filename matches what is required by the task
        subtasks = ['task2A', 'task2B']
        if not any(Path(pred_file_path).name.startswith(st_name) for st_name in subtasks):
            logging.error("The submission file must start by task name! possible prefixes: " + str(subtasks))
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files_path", "-p", nargs='+', required=True,
                        help="The absolute path to the files you want to check.", type=str)
    parser.add_argument("--subtask", "-s", required=True, choices=['2A', '2B'],
                        help="The subtask for which we are checking format: 2A, 2B", type=str)

    args = parser.parse_args()
    pred_files_path = args.pred_files_path
    subtask = args.subtask

    if validate_files(pred_files_path):
        for pred_file_path in pred_files_path:
            logging.info("Checking file: {}".format(pred_file_path))

            check_format(pred_file_path, subtask)
