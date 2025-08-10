# Data Description
## Set NLI dataset
A dataset created by modifying conventional pair-wise NLI dataset to become a set-based dataset. 
Each instance is a set of statements that are collectively either consistent or inconsistent with each other.
(The inconsistency may be due to two conflicting statements contained within the set, or arise from a combination of three or more statments.)
For rules used to create the dataset, please refer to the paper.

## Download link
https://drive.google.com/drive/folders/13nAyAR3b18iZ7kQQ2j7_c3qbqOAXt8y1?usp=sharing

(Place downloaded files in `datasets/set_nli/final`.)

## Data format
```json
    Output format
    {'pairID': '<pairID>', 
    'hypothesis': '...',
    'premise': '...',
    'hypothesis_negated': '...',
    'premise_negated': '...',
    'label': '...',
    'sets': [
        {'setID': '<pairID>_<no>',
        'set': [...],
        'label': '...',
        'set_size': ,
        'difficulty': '...',
        },...
    ], ...
    }
```
### Notes on difficulty
There are two levels of difficulty: medium and easy. 
Easy level is to distinguish inconsistent sets that include obvious contradiction pairs, i.e., p and not p.
All other cases are medium level. 

## Files and instance counts
The seed datatset used is SNLI test set. 
- `datasets/set_nli/final/set_nli_data_snli_test.json`
    - consistent sets : 113,686
        - counts by set size: {3: 46601, 2: 19955, 4: 37037, 5: 10093}
    - inconsistent sets : 56,843
        - counts by set size: {3: 19810, 2: 3377, 5: 13463, 4: 20193}

- `datasets/set_nli/final/set_nli_data_snli_test_full.json`
    - Note: The full dataset follows the same rules as the regular file. 
            However, when creating sets using two entailment seed pairs, 
            instead of selecting one random additional seed pair for each seed pair (thus using 3368 pairs of two seed pairs), 
            it considers all possible combinations of two from the available entailment pairs (thus use 5670028 (3368*3367/2) pairs of pairs).

    - consistent sets : 17,113,666
        - counts by set size: {3: 46652, 2: 19955, 4: 17036966, 5: 10093}
    - inconsistent sets : 17,056,823
        - counts by set size: {3: 19810, 2: 3377, 5: 17012917, 4: 20719}

## To Recreate
1. Download logicattack dataset for SNLI and place it under `datasets/logicattack`. From this data, we use negated premise,hypothesis, and some pre-created modifications of entailment pairs.
2. Make sure you have `datasets/set_nli/negation/snli_test_neutral_negated_clean.json` and `datasets/set_nli/negation/snli_test_neutral_negated_clean.json`. Or create files according to `datasets/set_nli/negation/README.md`. 
2. Run `datasets/set_nli/create_set_nli_data.py`. (`datasets/set_nli/create_set_nli_data.py --create_full` if want to create full dataset.)

