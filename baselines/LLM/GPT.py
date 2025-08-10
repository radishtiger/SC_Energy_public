import os, json, torch, itertools, time, re
import random
from openai import OpenAI

class GPT():
    def __init__(self, params):
        self.params = params
        if self.params['dataset'] == 'lconvqa':
            self.datapoint_type = "question-answer"
        elif self.params['dataset'] == 'set_nli':
            self.datapoint_type = 'sentence'
        self.few_shot_prompts_path = os.path.join("baselines", "LLM" , "few_shot_prompts.json")
        self.shot_num = self.params['baseline']['shot_num']
        self.prompt_template_for_prediction = None
        self.prompt_template_for_locate = None
        self.prediction_type = self.params['baseline']['prediction_type']
        if not 'do_not_initialize' in self.params['baseline']:
            self.initialize_prompt()
        
        self.openai_modelname_dict = {
            "gpt-4o-mini"       : "gpt-4o-mini-2024-07-18",
            "gpt-4o"            : "gpt-4o-2024-05-13",
            "gpt-4-turbo"       : "gpt-4-turbo-2024-04-09",
            "gpt-4"             : "gpt-4-0613",
            "gpt-3.5-turbo"     : "gpt-3.5-turbo-0125",
            "gpt-o1-mini"       : "o1-mini-2024-09-12",
            "gpt-o3-mini"       : "o3-mini-2025-01-31",
            "deepseek-r1"       : "deepseek-reasoner"
        }
        self.model_id = self.openai_modelname_dict.get(params['baseline']['model'].lower(), None)
        print(f"model: {self.model_id}")
        self.sanity_check()
        with open(os.path.join('private_key', 'key.json'), 'r') as f:
            private_key = json.load(f)

        if "gpt" in params['baseline']['model'].lower():
            self.client = OpenAI(api_key = private_key['OPENAI_API_KEY'],
                            #  organization = private_key.get('OPENAI_API_ORGANIZATION_ID', None)
                             )

    def api_call(self, prompt: str) -> str:
        
        if "o1" in self.model_id:
            messages=[
            # {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        else:
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model= self.model_id,
            messages=messages
        )
        return completion.choices[0].message.content.strip()

    def predict(self, pair):     
        if self.prediction_type == 'all_in_one':
            return self.predict_all_in_one(pair)
        
        elif self.prediction_type == 'one_to_one':
            return self.predict_one_to_one(pair)
        
        elif self.prediction_type == 'many_to_one':
            return self.predict_many_to_one(pair)
        
    def predict_all_in_one(self, pair):
        assert len(pair) == 1

        prompts = self.finalize_prompt(pair[0], mode = 'predict')
        print("prompt:")
        print(prompts)
        print()
        pred = self.api_call(prompts)
        print("api_call result:\n",pred)

        # transform to the integer
        pred = self.post_process_prediction(pred)
        if type(pred) == int:
            pred = [pred]

        print("prediction result:", pred)
        return pred
    
    def predict_debug(self, pair):
        """
        pair: list = [qa_pair_str1, qa_pair_str2, ...]
            example of "qa_pair_str": "question: is sky blue?, answer: yes"
        
        """
        self.prediction_type = "all_in_one"
        prompts = self.finalize_prompt([pair], mode = 'predict')
        # print("prompt:")
        # print(prompts)
        # print()
        pred_str = self.api_call(prompts)
        # print("prediction_str_result:")
        # print(pred_str)
        # print("=======")

        # transform to the integer
        pred_int = self.post_process_prediction(pred_str)

        return pred_int, pred_str
    
    def predict_one_to_one(self, pair):

        assert len(pair) == 1
        print("pair:", pair)

        pair_comb = list(itertools.combinations(pair[0][0], 2))

        for p in pair_comb:
            print("p:")
            print(p)
            print("-----")
            prompts = self.finalize_prompt([p], mode = 'predict')
            print("===\nprompt:")
            print(prompts)
            print("\n===\n")
            pred = self.api_call(prompts)
            print(f"pred:{pred}")

            # transform to the integer
            pred = self.post_process_prediction(pred)
            if type(pred) == int:
                pred = [pred]
            if pred == [1]:
                return pred
        

        return [0]
    
    def predict_many_to_one(self, pair):
        assert len(pair) == 1

        for i in range(len(pair[0][0])):
            p = {"premise": pair[0][0][:i] + pair[0][0][i+1:],
                 "hypothesis": pair[0][0][i]}
            # print(pair[0][:i] + pair[0][i+1:])
            # print(pair[0][i])
            prompts = self.finalize_prompt(p, mode = 'predict')
            # print("===\nprompt:")
            # print(prompts)
            # print("===\n")
            pred = self.api_call(prompts)
            # print(pred)

            # transform to the integer
            pred = self.post_process_prediction(pred)
            if type(pred) == int:
                pred = [pred]
            if pred == [1]:
                return pred

        # raise
        return [0]
    
    def locate(self, pair):
        prompts = self.finalize_prompt(pair[0], mode = 'locate')
        # print("prompt:")
        # print(prompts)
        # print()
        pred_str = self.api_call(prompts)
        # print(pred_str)
        # raise

        # transform to the integer
        pred_list = self.post_process_locate(pred_str)
        # print(pred_list)
        # raise
        if random.random() < 0.1:
            print("prompts:")
            print(prompts)
            print("pred_str:")
            print(pred_str)
            print("pred_list:")
            print(pred_list)
            

        return pred_list

    def wiqa_separate_inference(self, pairs):
        """
        Example of pairs (dictionary):
        {
            "paragraph": [
                "Water from oceans, lakes, swamps, rivers, and plants turns into water vapor",
                "Water vapor condenses into millions of tiny droplets that form clouds",
                "Clouds lose these droplets through rain or snow, also caused precipitation",
                "Precipitation is either absorbed into the ground or runs off into rivers",
                "Water that was absorbed into the ground is taken up by plants",
                "Plants lose water from their surfaces as vapor",
                "The vapor goes back into the atmosphere",
                "Water that runs off into rivers flows into ponds, lakes, or oceans",
                "The water evaporates back into the atmosphere",
                ""
            ],
            "choices": [
                {
                    "label": "A",
                    "text": "more"
                },
                {
                    "label": "B",
                    "text": "less"
                },
                {
                    "label": "C",
                    "text": "no effect"
                }
            ],
            "qa_pairs": [
                {
                    "question": "suppose during respiration happens, how will it affect there is less precipitation in the clouds.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose the weather is very mild happens, how will it affect there is less precipitation in the clouds.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose environment supportive of egg laying happens, how will it affect a less intense water cycle.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                },
                {
                    "question": "suppose less water for the seeds happens, how will it affect there will be less water vapor in the air.",
                    "answer_label": "no_effect",
                    "answer_label_as_choice": "C"
                }
            ]
        }
        """
        predictions = []

        instruction_para_template = "Given the following paragraphs, please select the correct answer for the given question. Return only the answer.\n"
        instruction_para_template += "Specifically, please carefully read the question. For example, assume a given question is 'suppose less water in the environment happens, how will it affect a less intense water cycle.'. If you think that less water in the environment occurs less intense water cycle, you need to answer as more because less water in the environment occured 'less' intense water cycle.\n"
        instruction_para_template += "Paragraphs:\n"

        for para in pairs["paragraph"]:
            instruction_para_template += f"{para}\n"

        choice_template = "\nChoices: [more, less]\n"

        for qa in pairs["qa_pairs"]:
            question = qa["question"]
            
            prompt = instruction_para_template + f"\nQuestion: {question}\n" + choice_template
            print("prompt:")
            print(prompt)

            result = str(self.api_call(prompt))
            print('result:')
            print(result)
            print("answer:")
            print(qa["answer_label"])
            print("=================\n\n")
            predictions.append(result)
        

        return predictions

            




    def initialize_prompt(self):
        """
        Generate a prompt-template used for prediction.

        Our prompt is composed of following four parts:
            (1) front_prompt: explain the user's intention
            (2) (optional) few_shot examples
            (3) input-output pairs (= what we want to evaluate)
            (4) end_prompt: 
                For prediction  : "Consistency: "
                For locate      : "Inconsistent examples: "

        This function aims to complete (1) and (2).
        """
        
        # (1) front_prompt
        tasks = ['prediction', 'locate']
        # tasks = ['prediction']
        self.prompt_template_for_prediction = f"Tell me whether the following {self.datapoint_type} pairs are consistent or inconsistent. \n"
        self.prompt_template_for_locate = f"Find the {self.datapoint_type} pairs among the following that are logically inconsistent with the rest. Specifically, identify the minimal collection of inconsistent pairs such that the remaining pairs are logically consistent with one another. If there are no inconsistent pairs, return nothing."
        self.prompt_template_for_prediction_and_locate = f"Your goal is to solve the following two tasks.\n First, {self.prompt_template_for_prediction} Second, {self.prompt_template_for_locate}"
        
        if self.shot_num == 0:
            return
        
        # (2) few-shot examples
        with open(self.few_shot_prompts_path, 'r') as f:
            prompt_dict = json.load(f)
        
        prompt_dicts_for_task = prompt_dict[self.params['dataset']]

        for t in tasks:
            prompt_dict_for_t= prompt_dicts_for_task[t][self.prediction_type] 
            prompt_list_for_t = [val for key, val in prompt_dict_for_t.items() if int(key) <= self.shot_num]
            if len(prompt_list_for_t) < self.shot_num:
                print(f"Number of few_shot examples is less than you want for task {t}.")
                print(f"Currently we have {len(prompt_list_for_t)}, while you want {self.shot_num}.")
                print(f"Nontheless, we don't raise any error. We utilize {len(prompt_list_for_t)} number of few-shot examples")
                self.shot_num = len(prompt_list_for_t)
                self.params['baseline']['shot_num'] = len(prompt_list_for_t)

            for i, prompt in enumerate(prompt_list_for_t):
                setattr(self, 
                        f"prompt_template_for_{t}", 
                        getattr(self, f"prompt_template_for_{t}") + f"\n[example {i+1}]\n{prompt}\n")
                # for example, if t == 'prediction', then this code executes:
                #   self['prompt_template_for_prediction'] += f"\n[example {i+1}]\n{prompt}\n"


    def sanity_check(self):
        if self.model_id == None:
            print("======")
            print("[Sanity Check Failed]")
            print(f"Invalid Model Name. Your name is {self.params['baseline']['model']}.")
            print("However, it must be one of:")
            for key, val in self.openai_modelname_dict.items():
                print(key)
                print(f"(which will be converted into {val})")
                print()
            print("======")

    def finalize_prompt(self, pairs, mode):
        prompts = ""
        if mode == 'predict':
            front_prompt = self.prompt_template_for_prediction
            if self.prediction_type == 'many_to_one':
                front_prompt += f"\nPlease let me know that whether the {self.datapoint_type} pairs in premise are logically consistent or inconsistent with the {self.datapoint_type} pair in hypothesis. Furthermore, if the {self.datapoint_type} pairs in premise is already logically inconsistent, then your answer should be inconsistent.\n"
            end_prompt = f"provide your consistency judgment by choosing either 'consistent' or 'inconsistent' After the 'Consistency:' mark"

            tmp = front_prompt + "\n [Problem]\n"
            if self.prediction_type in {"all_in_one", "one_to_one"}:
                for i, p in enumerate(pairs[0]):
                    tmp += f"({i+1}) {p}"
                tmp += " \n "
                tmp += end_prompt
                prompts = tmp
            elif self.prediction_type == 'many_to_one':
                tmp += "\n[Premise] "
                for i, premise in enumerate(pairs['premise']):
                    tmp += f"({i+1}) {premise}"
                    tmp += " \n "
                # print("tmp:", tmp)
                tmp += "[Hypothesis] "
                tmp += f"({len(pairs['premise'])+1}) {pairs['hypothesis']}\n"
                tmp += end_prompt
                prompts = tmp

        elif mode == 'locate':
            front_prompt = self.prompt_template_for_locate
            end_prompt = "Your response should only contain the numbers of the inconsistent pairs. \nInconsistent pairs:"
            
            tmp = front_prompt + "\n [Problem]\n"
            for i, p in enumerate(pairs[0]):
                tmp += f"({i+1}) {p}"
            tmp += " \n "
            tmp += end_prompt
            prompts = tmp


        # elif mode == 'prediction_and_locate':
        #     front_prompt = self.prompt_template_for_prediction_and_locate
        #     end_prompt = "You should give me two answers. For the first task, your answer should be only one word, either 'consistent' or 'inconsistent'.\nFor the second task, your answer should be only the number of inconsistent pair from the rest. \n\n Consistency: \n\n Inconsistent pairs: "

        else:
            print(f"Invalid mode here. Your mode is {mode}.")
            raise NotImplementedError
        

        # print("prompts:\n",prompts)
        # raise
        return prompts
    
    def post_process_prediction(self, prediction):
        # Not completed yet.

        prediction = prediction[prediction.rfind("Consistency:") + len("Consistency:"):]

        incon_detect = False
        con_detect = True
        if 'inconsistent' in prediction:
            incon_detect = True
        if ('consistent' in prediction):
            con_detect = True
            
        # if random.random() < 0.1:
            # print(f"prediction:", prediction)
            # print(f"con_detect, incon_detect: {con_detect}, {incon_detect}")
        if con_detect and (not incon_detect):
            return 0 # 0 means consistent
        elif incon_detect:
            return 1 # 1 means inconsistent
        else:
            return int(torch.randint(0, 2, (1,)))

    def post_process_locate(self, response_text):
        identifier = "[Inconsistent pairs]"
        text_index = response_text.find(identifier)
        text = response_text[text_index + len(identifier):]
        patterns = re.findall(r'(\d{1,2})', text)
        patterns = [int(p) for p in patterns]
        return patterns
