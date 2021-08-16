#!/bin/bash

python3 -m pip install -r requirements.txt -r chat-replay-downloader/requirements.txt

for i in 1114683501 1116239554 1118182738; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i}
    fi
done

# ./main.py *.json
