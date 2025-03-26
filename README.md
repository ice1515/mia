### Prerequisites
- Conda package manager

### Installation
1. Create the conda environment using the provided `environment.yml`:
   conda env create -f environment.yml
2. Activate the environment:
   conda activate gnn-decision-mia
3. Train all models using the shell script:
   sh code/train_model.sh 
   or 
   python main_train.py --config configs/TUs_graph_classification_GCN_DD.json --load 0 
4. Attack:
   sh code/attack.sh
   or
   python attack.py --config configs/TUs_graph_classification_GCN_DD.json --load 1  --noise_number 1000

  





