#!/bin/bash

# SSH parameters
HOST=" "
PORT=2222
REMOTE_DIR="/home/giveaway-6/arabic-nlp/task2/dat/dat/"
LOCAL_DIR="/home/slasher/araieval_arabicnlp24/task2/dat"

# JSON file path
JSON_FILE="$LOCAL_DIR/arabic_memes_propaganda_araieval_24_test.json"

# Counter for iterated files
count=0

# Read JSON file and iterate through objects
jq -c '.[]' "$JSON_FILE" | while read -r ROW; do
    # Increment counter
    ((count++))
    # If counter is less than 873, skip the iteration
    if [ $count -le 0 ]; then
        echo "Skip file $count"
        continue
    fi
    #if [ "$count" -eq 873 ] || [ "$count" -eq 1351 ]; then
    # Echo the current file number
    echo "Processing file $count"

    # Extract fields from JSON object
    ID=$(echo "$ROW" | jq -r '.id')
    SENTENCE=$(echo "$ROW" | jq -r '.text')
    IMAGE_PATH=$(echo "$ROW" | jq -r '.image_path')

    #scp -r -P $PORT $HOST:$REMOTE_DIR/$ID $LOCAL_DIR/$ID
    sshpass -p 'haouhat1' scp -r -P "$PORT" "$HOST:$REMOTE_DIR/${ID}_img.npy" "$LOCAL_DIR/${ID}_img.npy"
    if [ $? -eq 0 ]; then
        echo "File $ID copied successfully!"
    else
        echo "Error copying file ${HOST}:${REMOTE_DIR}/${ID}_img.npy"
    fi
    #fi
done
 echo "All files copied successfully!"
