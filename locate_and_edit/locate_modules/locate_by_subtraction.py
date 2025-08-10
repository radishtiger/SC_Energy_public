import re
import torch
import torch.nn as nn

class locate_by_subtraction():
    def __init__(self, params, energynet):
        self.params = params
        self.energynet = energynet
        self.softmax1 = nn.Softmax(dim = 1)
        self.recursive_locate = True # determined by self.initialize()
        self.sep_token = "."
        self.cls_token = self.energynet.representation_model.tokenizer.cls_token
        # self.initialize()

    def initialize(self):
        if 'recursive' in self.params['locate'].get('type', None).lower():
            self.recursive_locate = True
        else:
            self.recursive_locate = False

    def locate(self, pair, with_gold = True, threshold = None):
        # inputs = [p[0] for p in pair]
        # errors = [p[1] for p in pair]
        # print("inputs:")
        # print(inputs)
        # print("errors:")
        # print(errors)
        # raise
        if self.recursive_locate == False:
            locate_result = self.locate_only_once([p[0] for p in pair], with_gold, threshold)
        else:
            locate_result = self.locate_without_gold([p[0] for p in pair], threshold)

        error_gold = [p[1] for p in pair]
        
        pair_num = int(pair[0][0].count(self.sep_token))
        locate_result['errors_gold'] = error_gold # batch 1짜리 취급
        locate_result['pair_num'] = pair_num
        return locate_result

    def locate_only_once(self, pair, with_gold = True, threshold = None):
        result = {}
        if with_gold:
            inputs = [p[0] for p in pair]
            errors = [p[1] for p in pair]
            result['errors_gold'] = errors
        else:
            inputs = pair
        prediction_list = self.locate_without_gold(inputs, threshold)
        result['prediction_list'] = prediction_list
        return result
        
    def locate_without_gold(self, inputs, threshold = None):
        batch_size = len(inputs)

        prediction_list = []

        for b in range(batch_size):
            # print("[Area1]: original input_text:", inputs[b])
            spans = self.detect_span(inputs[b])
            # print("[Area2]: detected spans")
            # for s in spans:
                # print("\t",s)
            pairs_dict = {s:i for i,s in enumerate(spans)}

            inconsistent_pairs = self.locate_recursively(inputs[b], threshold, [])
            prediction_list.append([pairs_dict[p] for p in inconsistent_pairs])

        return {'prediction_list': prediction_list}
    
    def locate_recursively(self, long_text, threshold = None, previous_returns = []):
        result = {}
        
        # print("\n----")
        # print(f"stage {len(previous_returns)+1}: {long_text}")
        spans = self.detect_span(long_text)
        pair_num = len(spans)

        e_val, _ = self.energynet.energy_model([long_text], pair_only = True)
        if threshold == None:
            threshold = self.energynet.threshold
        
        if self.energynet.output_form == 'real_num':
            if float(e_val) < threshold or pair_num == 1:
                # print("[Ended] This is consistent long_text.\n\n")
                return previous_returns
        if self.energynet.output_form == '2dim_vec':
            if float(e_val[:,1]) < threshold or pair_num == 1:
                # print("[Ended] This is consistent long_text.\n\n")
                return previous_returns

        subtracted_inputs = self.subtract_input(spans)
        values = []

        # print("[Area3] subtracted inputs in recursive ft")
        # for s in subtracted_inputs:
            # print("\t"*(len(previous_returns)+1), s)
        
        for s_i in subtracted_inputs:

            if self.energynet.output_form == 'real_num':
                e_val, _ = self.energynet.energy_model([s_i], pair_only = True)
                values.append(float(e_val))
            if self.energynet.output_form == '2dim_vec':
                e_out, _ = self.energynet.energy_model([s_i], pair_only = True) # (len(set), 2)
                probs_for_incon = e_out[:, 1] # shape = (len(set))
                values.append(float(probs_for_incon))

        min_energy = min(values)
        prediction_int = list(values).index(min_energy)
            
        # print(f"prediction_int(index, starts from 0): {prediction_int}")
        prediction_string = spans[prediction_int]

        # print("[Area4]: predicted int and string + next string")
        # print("\t"*(len(previous_returns)+1), prediction_int)
        # print("\t"*(len(previous_returns)+1), prediction_string)
        # print("\t"*(len(previous_returns)+1), subtracted_inputs[prediction_int])


        if threshold == None:
            return [prediction_string]
        
        # elif min_energy < threshold or pair_num == 1:
        #     print(f"algorithm ended.")
        #     if min_energy < threshold:
        #         print("the remaining")
        #     return previous_returns + [prediction_string]
        
        else:
            new_pair = subtracted_inputs[prediction_int]
            return self.locate_recursively(new_pair, threshold, previous_returns + [prediction_string])
        

    def detect_span(self,set):

        # set == text, e.g., '<s> qa pair 1 </s> qa pair 2 ... </s>

        out = set[len(self.cls_token):].split(self.sep_token)[:-1]
        
        return [o.strip()+self.sep_token for o in out]
    
    def subtract_input(self, spans):

        subtracted_list = []
        for i, s in enumerate(spans):

            sub_string = f"{self.cls_token} "
            sub_string += ' '.join([spans[j] for j in range(len(spans)) if j !=i])

            subtracted_list.append(sub_string)

        return subtracted_list