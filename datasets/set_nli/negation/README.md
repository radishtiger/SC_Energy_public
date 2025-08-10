# Datasets created by running preprocess_snli_data.py on the snli_1.0_test.txt
## What the script does : use gpt4o-mini to create negated versions of premises and hypotheses
- tasks/nli/negation/snli_test_contradiction_negated.json : 3237 contradiction-labelled examples with negated premise & hypothesis
- tasks/nli/negation/snli_test_neutral_negated.json : 3219 neutral-labelled examples with negated premise & hypothesis

c.f. We didn't create negations for entailment pairs since logicattack dataset (https://github.com/msantoshmadhav/LogicAttack) provides negations for them.

## Regarding _clean datasets
We recommend using _clean datasets.

- tasks/nli/negation/snli_test_contradiction_negated_clean.json : 
    - 123 (122+1) negations corrected
    - 1 instance dropped (515664903.jpg#2r1c) b/c the original hypothesis was erroneous. ("Thi")
- tasks/nli/negation/snli_test_neutral_negated_clean.json : 81 negations corrected

Upon inspection, some of the negations contained a type of error where "The \<subject\> \<verb\>" is negated as "No \<subject\> \<verb\>" while the correct negation is "The \<subject\> \<negated verb\>"
Thus, we created a "clean" version of datasets by manually correcting negations with the error.
We also found other cases of errors and corrected them as well.
