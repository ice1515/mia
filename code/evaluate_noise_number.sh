
#!/bin/bash

noise_number="1 2 3 4 5 6 7 8 9 10 50 100 200 500 1000"
config_file="configs/TUs_graph_classification_GraphSage_DD.json"

for number in $noise_number; do
    echo "======================================================================"
    echo "Running: noise number: ${number}"
    python attack.py \
        --config ${config_file} \
        --load 1 \
        --noise_number ${number}\
        --attack_num 5
done

