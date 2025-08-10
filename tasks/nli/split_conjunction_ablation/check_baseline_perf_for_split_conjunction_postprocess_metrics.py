from glob import glob
import numpy as np

pred_proba_files = glob('/data/hyeryung/set_consistency_energy/tasks/nli/split_conjunction_ablation/*/*/predictions.txt')
result_file = open('/data/hyeryung/set_consistency_energy/tasks/nli/split_conjunction_ablation/performances_expanded.csv', 'w')
result_file.write('nli_model,data_loader_type,accuracy_entail,accuracy_consistent,avg_entail_proba,avg_consistent_proba\n')

for pred_proba_file in pred_proba_files:
    
    model_name = pred_proba_file.split('/')[-3]
    dataloader_type = pred_proba_file.split('/')[-2]
    print(model_name, dataloader_type)
    
    with open(pred_proba_file, 'r') as f:
        lines = [x.strip().split(',') for x in f.readlines()]
    data = np.array(lines).astype(np.float32)
    if 'cross_encoder' in model_name:
        entail_col,contra_col = 1,0
    else:
        entail_col,contra_col = 0,2
        
    avg_entail_proba = np.mean(data[:, entail_col])
    avg_consistent_proba = np.mean(1 - data[:, contra_col])
    accuracy_entail = np.mean(data.argmax(axis=-1) == entail_col)
    accuracy_consistent = np.mean(data.argmax(axis=-1) != contra_col)
    result_file.write(f'{model_name},{dataloader_type},{accuracy_entail:.4f},{accuracy_consistent:.4f},{avg_entail_proba:.4f},{avg_consistent_proba:.4f}\n')

result_file.close()