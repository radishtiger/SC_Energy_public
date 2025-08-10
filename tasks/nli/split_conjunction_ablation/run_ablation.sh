#!/bin/bash
# dl_types=('regular' 'split_conjunction')
dl_types=('regular_with_and' 'logic_attack')
# nli_models=('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
nli_models=('cross-encoder/nli-deberta-v3-base' 'cross-encoder/nli-roberta-base' 'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli' 'tasksource/deberta-base-long-nli')
for dl_type in ${dl_types[@]}; do
    for nli_model in ${nli_models[@]}; do
        python tasks/nli/check_baseline_perf_for_split_conjunction.py \
            --dataloader_type ${dl_type} \
            --nli_model ${nli_model}
    done
done