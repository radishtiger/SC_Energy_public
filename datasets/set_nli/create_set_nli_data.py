import json
import itertools
import random
from copy import deepcopy
import argparse

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def format_text(input_text):
    input_text = input_text[0].upper() + input_text[1:]
    input_text += '.'
    return input_text

def preprocess_no_decomposition(input_set):
    output_text = '<s>'
    for statement in input_set:
        output_text += statement
        output_text += '</s>'
        
    return output_text

def gen_entailment(sentence1, sentence2):
    sentence1 = sentence1.lower().rstrip('.')
    sentence2 = sentence2.lower().rstrip('.')
    return f"If {sentence1}, then {sentence2}."

def gen_or(sentence1, sentence2):
    sentence1 = sentence1.lower().rstrip('.')
    sentence2 = sentence2.lower().rstrip('.')
    return f"Either {sentence1} or {sentence2}."

def gen_and(sentence):
    sentence = sentence.lower().rstrip('.')
    return f"And {sentence}."

def create_single_entailment_seed_pair_sets(data_instance: dict):
    
    sample_p = data_instance['premise']
    sample_h = data_instance['hypothesis']
    sample_p_neg = data_instance['premise_negated']
    sample_h_neg = data_instance['hypothesis_negated']

    sample_transportation_p = data_instance['attack_samples'][5]['premise']
    sample_transportation_h = data_instance['attack_samples'][5]['hypothesis']

    sample_material_imp_p = data_instance['attack_samples'][7]['premise']
    sample_material_imp_h = data_instance['attack_samples'][7]['hypothesis']

    sample_material_imp_half_1_p = data_instance['attack_samples'][7]['premise']
    sample_material_imp_half_1_h = format_text(sample_h)

    sample_material_imp_half_2_p = data_instance['attack_samples'][7]['premise']
    sample_material_imp_half_2_h = format_text(sample_p_neg)
    
    sample_p_or_h = gen_or(sample_p, sample_h)

    nli_sets_v2_con = [
        [{sample_transportation_p, format_text(sample_p), format_text(sample_h)}, 'consistent', 'medium'], # modus ponens
        [{sample_transportation_p, format_text(sample_h_neg), format_text(sample_p_neg)}, 'consistent', 'medium'], # modus tollens
        [{sample_transportation_p, sample_transportation_h}, 'consistent', 'medium'], # transportation
        [{sample_material_imp_p, sample_material_imp_h}, 'consistent', 'medium'], # material implication    
        [{sample_material_imp_half_1_p, sample_material_imp_half_1_h}, 'consistent', 'medium'], # derived from material implication (i.e., material implication's premise first half)
        [{sample_material_imp_half_2_p, sample_material_imp_half_2_h}, 'consistent', 'medium'], # derived from material implication (i.e., material implication's premise second half)
        [{sample_p_or_h, sample_h_neg, sample_p}, 'consistent', 'medium'],
        [{sample_p_or_h, sample_p_neg, sample_h}, 'consistent', 'medium'],
    ]

    nli_sets_v2_incon = [
        [{sample_transportation_p, format_text(sample_p), format_text(sample_h_neg)}, 'inconsistent', 'medium'], # negate hypothesis of modus ponens
        [{format_text(sample_p), format_text(sample_h_neg)}, 'inconsistent','medium'], # negate hypothesis of original pair
        [{gen_or(sample_p, sample_h), sample_p_neg, sample_h_neg}, 'inconsistent','medium'], # negate either side of disjunction
        [{gen_or(sample_p, sample_h), gen_entailment(sample_p, sample_h), sample_h_neg}, 'inconsistent','medium'], # negate hypothesis of disjunction when premise entails hypothesis
    ]

    return nli_sets_v2_con, nli_sets_v2_incon

def augment_consistent_sets(nli_sets_v2_con):
    
    num_seed_sets = 6 ## set as 6 because we won't use disjunctive syllogism for augmenting
    
    # Choose 2 among 6 sets and drop 3 inconsistent sets and 2 duplicate sets
    nli_sets_v2_con_c2 = []
    for i in range(0, num_seed_sets):
        for j in range(i+1, num_seed_sets):
            if ((i==0) and (j==1)) or ((i==0) and (j==5)) or ((i==1) and (j==4)): # inconsistent sets
                continue
            elif ((i==0) and (j==4)) or ((i==1) and (j==5)): # duplicate sets
                continue
            else:
                nli_sets_v2_con_c2.append([nli_sets_v2_con[i][0].union(nli_sets_v2_con[j][0]), 'consistent', 'medium'])

    # Choose 3 among 6 sets and drop 10 inconsistent sets and 4 duplicate sets
    nli_sets_v2_con_c3 = []
    for i in range(0, num_seed_sets):
        for j in range(i+1, num_seed_sets):
            for k in range(j+1, num_seed_sets):
                if ((0 in [i,j,k]) and ((1 in [i, j, k]) or (5 in [i, j, k]))) or \
                   ((1 in [i,j,k]) and ((4 in [i, j, k]))): # inconsistent sets
                    continue
                elif ((0 in [i, j, k]) and (4 in [i,j,k])) or ((1 in [i, j, k]) and (5 in [i,j,k])): # duplicate sets
                    continue
                else:
                    nli_sets_v2_con_c3.append([nli_sets_v2_con[i][0].union(nli_sets_v2_con[j][0]).union(nli_sets_v2_con[k][0]), 'consistent', 'medium'])

    # Choose 4 among 6 sets and drop 12 inconsistent sets and 2 duplicate sets
    nli_sets_v2_con_c4 = [[nli_sets_v2_con[2][0].union(nli_sets_v2_con[3][0]).union(nli_sets_v2_con[4][0]).union(nli_sets_v2_con[5][0]), 'consistent', 'medium']]

    return nli_sets_v2_con + nli_sets_v2_con_c2 + nli_sets_v2_con_c3 + nli_sets_v2_con_c4

def augment_inconsistent_sets(nli_sets_v2_con, nli_sets_v2_incon):
    # Combine 2 simple consistent sets that contradict each other
    nli_sets_v2_incon_c2 = []
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[0][0].union(nli_sets_v2_con[1][0]), 'inconsistent', 'easy']] # modus tollens & modus ponens
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[1][0].union(nli_sets_v2_con[4][0]), 'inconsistent', 'easy']] # modus tollens & derived from material implication (i.e., material implication's premise first half; h)
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[0][0].union(nli_sets_v2_con[5][0]), 'inconsistent', 'easy']] # modus ponens & derived from material implication (i.e., material implication's premise second half; p neg)

    # Combine 1 simple inconsistent case (#B.) with 1 simple consistent case
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[0][0].union(nli_sets_v2_incon[0][0]), 'inconsistent', 'easy']] # modus ponens & rule 1
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[1][0].union(nli_sets_v2_incon[0][0]), 'inconsistent', 'easy']] # modus tollens & rule 1
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[2][0].union(nli_sets_v2_incon[0][0]), 'inconsistent', 'medium']] # transportation & rule 1
    nli_sets_v2_incon_c2 += [[nli_sets_v2_con[3][0].union(nli_sets_v2_incon[0][0]), 'inconsistent', 'medium']] # material implication & rule 1

    return nli_sets_v2_incon + nli_sets_v2_incon_c2

def create_single_contradiction_seed_pair_sets(data_instance):
    
    sample_p = format_text(data_instance['premise'])
    sample_h = format_text(data_instance['hypothesis'])
    sample_p_neg = format_text(data_instance['premise_negated'])
    sample_h_neg = format_text(data_instance['hypothesis_negated'])
    
    sample_p_or_h = gen_or(sample_p, sample_h)
    
    nli_sets_v2_incon = []
    nli_sets_v2_incon += [[{sample_p_or_h, sample_p, sample_h}, 'inconsistent', 'medium'],
                        [{sample_p_or_h, sample_p_neg, sample_h_neg}, 'inconsistent', 'medium']]
    
    nli_sets_v2_con = []
    nli_sets_v2_con += [[{sample_p_neg, sample_h}, 'consistent', 'medium'],
                        [{sample_p, sample_h_neg}, 'consistent', 'medium'],
                        [{sample_p_or_h, sample_h_neg, sample_p}, 'consistent', 'medium'],
                        [{sample_p_or_h, sample_p_neg, sample_h}, 'consistent', 'medium'],]
    
    return nli_sets_v2_con, nli_sets_v2_incon
    
def create_single_neutral_seed_pair_sets(data_instance):
    
    sample_p = format_text(data_instance['premise'])
    sample_h = format_text(data_instance['hypothesis'])
    sample_p_neg = format_text(data_instance['premise_negated'])
    sample_h_neg = format_text(data_instance['hypothesis_negated'])
    
    sample_p_or_h = gen_or(sample_p, sample_h)
    
    nli_sets_v2_incon = []
    nli_sets_v2_incon += [[{sample_p_or_h, sample_p_neg, sample_h_neg}, 'inconsistent', 'medium']]
    
    nli_sets_v2_con = []
    nli_sets_v2_con += [[{sample_p_or_h, sample_h_neg, sample_p}, 'consistent', 'medium'],
                        [{sample_p_or_h, sample_p_neg, sample_h}, 'consistent', 'medium'],]
    
    return nli_sets_v2_con, nli_sets_v2_incon

def create_two_seed_pairs_sets(data_instance, second_data_instance):
    sample_p = data_instance['premise']
    sample_h = data_instance['hypothesis']
    sample_p_neg = data_instance['premise_negated']
    sample_h_neg = data_instance['hypothesis_negated']

    sample_p2 = second_data_instance['premise']
    sample_h2 = second_data_instance['hypothesis']
    sample_p2_neg = second_data_instance['premise_negated']
    sample_h2_neg = second_data_instance['hypothesis_negated']

    sample_const_dilemma_p = f"{gen_entailment(sample_p, sample_h)} {gen_and(gen_entailment(sample_p2, sample_h2))} {gen_and(gen_or(sample_p, sample_p2))}"
    sample_const_dilemma_h = gen_or(sample_h, sample_h2)
    sample_const_dilemma_h_neg = f"{format_text(sample_h_neg)} {gen_and(sample_h2_neg)}"

    sample_dest_dilemma_p = f"{gen_entailment(sample_p, sample_h)} {gen_and(gen_entailment(sample_p2, sample_h2))} {gen_and(gen_or(sample_h_neg, sample_h2_neg))}"
    sample_dest_dilemma_h = gen_or(sample_p_neg, sample_p2_neg)
    sample_dest_dilemma_h_neg = f"{format_text(sample_p)} {gen_and(sample_p2)}"

    sample_bidir_dilemma_p = f"{gen_entailment(sample_p, sample_h)} {gen_and(gen_entailment(sample_p2, sample_h2))} {gen_and(gen_or(sample_p, sample_h2_neg))}"
    sample_bidir_dilemma_h = gen_or(sample_h, sample_p2_neg)
    sample_bidir_dilemma_h_neg = f"{format_text(sample_h_neg)} {gen_and(sample_p2)}"

    sample_entail_1 = gen_entailment(sample_p, sample_h)
    sample_entail_2 = gen_entailment(sample_p2, sample_h2)
    sample_p_or = gen_or(sample_p, sample_p2)
    sample_p_or_neg = f"{format_text(sample_p_neg)} {gen_and(format_text(sample_p2_neg))}"
    sample_h_or = gen_or(sample_h, sample_h2)
    sample_h_or_neg = f"{format_text(sample_h_neg)} {gen_and(format_text(sample_h2_neg))}"
    sample_h_neg_or = gen_or(sample_h_neg, sample_h2_neg)
    sample_h_neg_or_neg = f"{format_text(sample_h)} {gen_and(format_text(sample_h2))}"
    sample_p_neg_or = gen_or(sample_p_neg, sample_p2_neg)
    sample_p_neg_or_neg = f"{format_text(sample_p)} {gen_and(format_text(sample_p2))}"
    sample_p_h2_neg_or = gen_or(sample_p, sample_h2_neg)
    sample_p_h2_neg_or_neg = f"{format_text(sample_p_neg)} {gen_and(format_text(sample_h2))}"
    sample_h_p2_neg_or = gen_or(sample_h, sample_p2_neg)
    sample_h_p2_neg_or_neg = f"{format_text(sample_h_neg)} {gen_and(format_text(sample_p2))}"

    nli_sets_s2_v2_con = [
        [{sample_entail_1, sample_entail_2, sample_p_or, sample_h_or}, 'consistent', 'medium'], # constructive dilemma
        [{sample_entail_1, sample_entail_2, sample_h_neg_or, sample_p_neg_or}, 'consistent', 'medium'], # destructive dilemma
        [{sample_entail_1, sample_entail_2, sample_p_h2_neg_or, sample_h_p2_neg_or}, 'consistent', 'medium'], # bidirectional dilemma
    ]
    nli_sets_s2_v2_incon = [
        [{sample_entail_1, sample_entail_2, sample_p_or, format_text(sample_h_neg), format_text(sample_h2_neg)}, 'inconsistent', 'medium'], # negate hypothesis of constructive dilemma               
        [{sample_entail_1, sample_entail_2, sample_h_neg_or, format_text(sample_p), format_text(sample_p2)}, 'inconsistent', 'medium'], # negate hypothesis of destructive dilemma
        [{sample_entail_1, sample_entail_2, sample_p_h2_neg_or, format_text(sample_h_neg), format_text(sample_p2)}, 'inconsistent', 'medium'], # negate hypothesis of bidirectional dilemma
    ]

    # Not using these rules for this paper
    # # Augment consistent sets
    # # Among total 6 unique "atomic" statements apart from entailments (p1 → h1 and p2 → h2) existing in 3 Simple Cases, 
    # # choose 1 to 6 and combine with entailments (p1 → h1 and p2 → h2)
    # common_pairs = {sample_entail_1, sample_entail_2}
    # candidate_pairs = [sample_p_or, sample_h_or, 
    #                    sample_h_neg_or, sample_p_neg_or, sample_p_h2_neg_or, sample_h_p2_neg_or]
    
    # nC1 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 1)]
    # nC2 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 2)]
    # nC2_drop_dup = [_set for _set in nC2 if _set[0] not in [s[0] for s in nli_sets_s2_v2_con]] # *Drop 3 sets duplicate with 3 simple cases
    # nC3 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 3)]
    # nC4 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 4)]
    # nC5 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 5)]
    # nC6 = [[common_pairs.union(set(x)), 'consistent', 'medium'] for x in itertools.combinations(candidate_pairs, 6)]

    # final_nli_sets_s2_v2_con = nli_sets_s2_v2_con + nC1 + nC2_drop_dup + nC3 + nC4 + nC5 + nC6

    # # Augment inconsistent sets
    # # Combine hypothesis negation of constructive dilemma with 4C{1,2,3,4} sets from total 4 unique "atomic" statements that are not part of hypothesis negation of constructive dilemma.
    # contradict_pairs = {sample_p_or, sample_h_or_neg}
    # other_candidate_pairs = [sample_p_neg_or, sample_h_neg_or, sample_p_h2_neg_or, sample_h_p2_neg_or]

    # nC1 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 1)]
    # nC2 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 2)]
    # nC3 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 3)]
    # nC4 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 4)]

    # nli_sets_s2_incon_const_dil_mod = nC1 + nC2 + nC3 + nC4

    # # Combine hypothesis negation of destructive dilemma with 4C{1,2,3,4} sets from total 4 unique "atomic" statements that are not part of hypothesis negation of destructive dilemma.
    # contradict_pairs = {sample_h_neg_or, sample_p_neg_or_neg}
    # other_candidate_pairs = [sample_p_or, sample_h_or, sample_p_h2_neg_or, sample_h_p2_neg_or]

    # nC1 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 1)]
    # nC2 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 2)]
    # nC3 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 3)]
    # nC4 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 4)]

    # nli_sets_s2_incon_dest_dil_mod = nC1 + nC2 + nC3 + nC4

    # # Combine hypothesis negation of bidirectional dilemma with 4C{1,2,3,4} sets from total 4 unique "atomic" statements that are not part of hypothesis negation of bidirectional dilemma.
    # contradict_pairs = {sample_p_h2_neg_or, sample_h_p2_neg_or_neg}
    # other_candidate_pairs = [sample_p_or, sample_h_or, sample_p_neg_or, sample_h_neg_or]

    # nC1 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 1)]
    # nC2 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 2)]
    # nC3 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 3)]
    # nC4 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'medium'] for x in itertools.combinations(other_candidate_pairs, 4)]

    # nli_sets_s2_incon_bidir_dil_mod = nC1 + nC2 + nC3 + nC4

    # # Add one complex inconsistent set that is ¬ (p1 ∨ ¬ h2), (p1 ∨ p2),  (h2 ∨ ¬ p2)
    # contradict_pairs = {sample_p_h2_neg_or_neg, sample_p_or, sample_h_p2_neg_or}
    # other_candidate_pairs = [sample_h_or, sample_p_neg_or, sample_h_neg_or]
    # nC0 = [[common_pairs.union(contradict_pairs), 'inconsistent', 'hard']]
    # # And then combine ¬ (p1 ∨ ¬ h2), (p1 ∨ p2),  (h2 ∨ ¬ p2) with 3C{1,2,3} sets from total 4 unique "atomic" statements that are not part of ¬ (p1 ∨ ¬ h2), (p1 ∨ p2),  (h2 ∨ ¬ p2).
    # nC1 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'hard'] for x in itertools.combinations(other_candidate_pairs, 1)]
    # nC2 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'hard'] for x in itertools.combinations(other_candidate_pairs, 2)]
    # nC3 = [[common_pairs.union(contradict_pairs).union(set(x)), 'inconsistent', 'hard'] for x in itertools.combinations(other_candidate_pairs, 3)]

    # nli_sets_s2_incon_sample_p_h2_neg_or_neg_mod = nC0 + nC1 + nC2 + nC3

    # final_nli_sets_s2_v2_incon = nli_sets_s2_v2_incon + nli_sets_s2_incon_const_dil_mod + \
    #     nli_sets_s2_incon_dest_dil_mod + nli_sets_s2_incon_bidir_dil_mod + \
    #     nli_sets_s2_incon_sample_p_h2_neg_or_neg_mod

    # return final_nli_sets_s2_v2_con, final_nli_sets_s2_v2_incon
    return nli_sets_s2_v2_con, nli_sets_s2_v2_incon

def get_size_count(result_set):
    size_cnt = {}
    for example in result_set:
        if len(example[0]) not in size_cnt:
            size_cnt[len(example[0])] = 1
        else:
            size_cnt[len(example[0])] += 1
    return size_cnt

def main(create_full=False):
    # Load data
    data = load_data('datasets/logicattack/logic_attacks_snli_test.json')
    data_contradict = load_data('datasets/set_nli/negation/snli_test_contradiction_negated_clean.json')
    data_neutral = load_data('datasets/set_nli/negation/snli_test_neutral_negated_clean.json')

    # Output format
    # {'pairID': '...', 
    # 'hypothesis': '...',
    # 'premise': '...',
    # 'hypothesis_negated': '...',
    # 'premise_negated': '...',
    # 'label': '...',
    # 'sets': [
        # {'setID': '<pairID>_<no>',
        # 'set': [...],
        # 'label': '...',
        # 'set_size': %d,
        # 'difficulty': '...',
        # },...
    # ], ...
    # }
    
    
    result_set_con = []
    result_set_incon = []
    result = []
    
    # Create single seed pair sets
    for pair in data['instances']:
        
        nli_sets_v2_con, nli_sets_v2_incon = create_single_entailment_seed_pair_sets(pair)
        # print(f"# consistent sets (single seed pair) before augmentation: {len(nli_sets_v2_con)}")
        # print(f"# inconsistent sets (single seed pair) before augmentation: {len(nli_sets_v2_incon)}")
        
        # Augment consistent and inconsistent sets
        final_nli_sets_v2_con = augment_consistent_sets(nli_sets_v2_con)
        final_nli_sets_v2_incon = augment_inconsistent_sets(nli_sets_v2_con, nli_sets_v2_incon)
        
        result_set_con += final_nli_sets_v2_con
        result_set_incon += final_nli_sets_v2_incon

        # print(f"# consistent sets (single seed pair): {len(final_nli_sets_v2_con)}")
        # print(f"# inconsistent sets (single seed pair): {len(final_nli_sets_v2_incon)}")
        
        pair_for_write = deepcopy(pair)
        del pair_for_write['attack_samples']
        
        pair_for_write['sets'] = []
        
        cnt = 0
        for item in final_nli_sets_v2_con + final_nli_sets_v2_incon:
            pair_for_write['sets'].append({'setID': f"{pair['pairID']}_{cnt}",
                                            'set': list(item[0]),
                                            'label': item[1],
                                            'set_size': len(list(item[0])),
                                            'difficulty': item[2]})
            cnt += 1
            
        result.append(pair_for_write)
        
    
    print('-'*50)
    print('entail seed (single pair)')
    print('cons')
    print(get_size_count(result_set_con))
    print('incons')
    print(get_size_count(result_set_incon))
    
    # Create and augment two seed pairs sets
    if create_full: 
        for i in range(len(data['instances'])):
            for j in range(i+1, len(data['instances'])):
            # for _ in range(1):
                # indexes_except_i = list(set(range(len(data['instances']))) - {i})
                # j = random.choice(indexes_except_i)
                final_nli_sets_s2_v2_con, final_nli_sets_s2_v2_incon = create_two_seed_pairs_sets(data['instances'][i], data['instances'][j])
            
                result_set_con += final_nli_sets_s2_v2_con
                result_set_incon += final_nli_sets_s2_v2_incon

                # print(f"# consistent sets (two seed pairs): {len(final_nli_sets_s2_v2_con)}")
                # print(f"# inconsistent sets (two seed pairs): {len(final_nli_sets_s2_v2_incon)}")
                
                pair_for_write = {
                    "pairID_1": data['instances'][i]["pairID"],
                    "premise_1": data['instances'][i]["premise"],
                    "hypothesis_1": data['instances'][i]["hypothesis"],
                    "premise_negated_1": data['instances'][i]["premise_negated"],
                    "hypothesis_negated_1": data['instances'][i]["hypothesis_negated"],
                    "label_1": data['instances'][i]["label"],
                    "pairID_2": data['instances'][j]["pairID"],
                    "premise_2": data['instances'][j]["premise"],
                    "hypothesis_2": data['instances'][j]["hypothesis"],
                    "premise_negated_2": data['instances'][j]["premise_negated"],
                    "hypothesis_negated_2": data['instances'][j]["hypothesis_negated"],
                    "label_2": data['instances'][j]["label"],
                    "sets": [],
                }
                cnt = 0
                for item in final_nli_sets_s2_v2_con + final_nli_sets_s2_v2_incon:
                    pair_for_write['sets'].append({'setID': f"{pair['pairID']}_{cnt}",
                                                    'set': list(item[0]),
                                                    'label': item[1],
                                                    'set_size': len(list(item[0])),
                                                    'difficulty': item[2]})
                    cnt += 1
                    
                result.append(pair_for_write)
    else:
        for i in range(len(data['instances'])):
            for _ in range(1):
                indexes_except_i = list(set(range(len(data['instances']))) - {i})
                j = random.choice(indexes_except_i)
                final_nli_sets_s2_v2_con, final_nli_sets_s2_v2_incon = create_two_seed_pairs_sets(data['instances'][i], data['instances'][j])
            
                result_set_con += final_nli_sets_s2_v2_con
                result_set_incon += final_nli_sets_s2_v2_incon

                # print(f"# consistent sets (two seed pairs): {len(final_nli_sets_s2_v2_con)}")
                # print(f"# inconsistent sets (two seed pairs): {len(final_nli_sets_s2_v2_incon)}")
                
                pair_for_write = {
                    "pairID_1": data['instances'][i]["pairID"],
                    "premise_1": data['instances'][i]["premise"],
                    "hypothesis_1": data['instances'][i]["hypothesis"],
                    "premise_negated_1": data['instances'][i]["premise_negated"],
                    "hypothesis_negated_1": data['instances'][i]["hypothesis_negated"],
                    "label_1": data['instances'][i]["label"],
                    "pairID_2": data['instances'][j]["pairID"],
                    "premise_2": data['instances'][j]["premise"],
                    "hypothesis_2": data['instances'][j]["hypothesis"],
                    "premise_negated_2": data['instances'][j]["premise_negated"],
                    "hypothesis_negated_2": data['instances'][j]["hypothesis_negated"],
                    "label_2": data['instances'][j]["label"],
                    "sets": [],
                }
                cnt = 0
                for item in final_nli_sets_s2_v2_con + final_nli_sets_s2_v2_incon:
                    pair_for_write['sets'].append({'setID': f"{pair['pairID']}_{cnt}",
                                                    'set': list(item[0]),
                                                    'label': item[1],
                                                    'set_size': len(list(item[0])),
                                                    'difficulty': item[2]})
                    cnt += 1
                    
                result.append(pair_for_write)
        

    print('-'*50)
    print('entail seed (two pairs)')
    print('cons')
    print(get_size_count(final_nli_sets_s2_v2_con))
    print('incons')
    print(get_size_count(final_nli_sets_s2_v2_incon))

    
    for pair in data_contradict['instances']:
        nli_sets_v2_con, nli_sets_v2_incon = create_single_contradiction_seed_pair_sets(pair)
        result_set_con += nli_sets_v2_con
        result_set_incon += nli_sets_v2_incon
        # print(f"# consistent sets (single contra pair): {len(nli_sets_v2_con)}")
        # print(f"# inconsistent sets (single contra pair): {len(nli_sets_v2_incon)}")
        
        pair_for_write = deepcopy(pair)
        
        pair_for_write['sets'] = []
        
        cnt = 0
        for item in nli_sets_v2_con + nli_sets_v2_incon:
            pair_for_write['sets'].append({'setID': f"{pair['pairID']}_{cnt}",
                                            'set': list(item[0]),
                                            'label': item[1],
                                            'set_size': len(list(item[0])),
                                            'difficulty': item[2]})
            cnt += 1
            
        result.append(pair_for_write)
        
    
    print('-'*50)
    print('contradiction seed')
    print('cons')
    print(get_size_count(nli_sets_v2_con))
    print('incons')
    print(get_size_count(nli_sets_v2_incon))
    
    for pair in data_neutral['instances']:
        nli_sets_v2_con, nli_sets_v2_incon = create_single_neutral_seed_pair_sets(pair)
        result_set_con += nli_sets_v2_con
        result_set_incon += nli_sets_v2_incon
        # print(f"# consistent sets (single neutral pair): {len(nli_sets_v2_con)}")
        # print(f"# inconsistent sets (single neutral pair): {len(nli_sets_v2_incon)}")
        
        pair_for_write = deepcopy(pair)
        
        pair_for_write['sets'] = []
        
        cnt = 0
        for item in nli_sets_v2_con + nli_sets_v2_incon:
            pair_for_write['sets'].append({'setID': f"{pair['pairID']}_{cnt}",
                                            'set': list(item[0]),
                                            'label': item[1],
                                            'set_size': len(list(item[0])),
                                            'difficulty': item[2]})
            cnt += 1
            
        result.append(pair_for_write)
        
    
    print('-'*50)
    print('neutral seed')
    print('cons')
    print(get_size_count(nli_sets_v2_con))
    print('incons')
    print(get_size_count(nli_sets_v2_incon))

    print('-'*50)
    print('total_count')
    print('cons')
    print(get_size_count(result_set_con))
    print('incons')
    print(get_size_count(result_set_incon))
    
    
    if create_full:
        fname = 'datasets/set_nli/final/set_nli_data_snli_test_full.json'
    else:
        fname = 'datasets/set_nli/final/set_nli_data_snli_test.json'
    with open(fname, 'w') as f:
        json.dump({'dataset_name': 'SNLI', 
                   'instances': result}, f, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_full', action='store_true')
    args = parser.parse_args()
    
    main(create_full=args.create_full)