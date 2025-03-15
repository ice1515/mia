python attack.py --config configs/TUs_graph_classification_GAT_DD.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GCN_DD.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GraphSage_DD.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GIN_DD.json --load 1  --noise_number 1000



python attack.py --config configs/TUs_graph_classification_GAT_ENZYMES.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GraphSage_ENZYMES.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GCN_ENZYMES.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GIN_ENZYMES.json --load 1  --noise_number 1000


python attack.py --config configs/TUs_graph_classification_GAT_PROTEINS_full.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GraphSage_PROTEINS_full.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GCN_PROTEINS_full.json --load 1  --noise_number 1000
python attack.py --config configs/TUs_graph_classification_GIN_PROTEINS_full.json --load 1  --noise_number 1000





