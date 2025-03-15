scaler_lst="3"

config_file="configs/TUs_graph_classification_GAT_ENZYMES.json"

for scaler in $scaler_lst; do
    echo "======================================================================"
    echo "Running: noise number: ${scaler}"
    python attack_scaler.py \
        --config ${config_file} \
        --load 1 \
        --noise_number 1000 \
        --attack_num 5\
        --estimate_ratio  0\
        --scaler $scaler

done