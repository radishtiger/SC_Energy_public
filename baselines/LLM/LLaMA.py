import os, json, torch
import transformers

class LLaMA():
    def __init__(self, params):
        self.params = params
        self.few_show_prompts_path = os.path.join("baselines", "LLM" , "few_shot_prompts.json")
        self.shot_num = self.params['baseline']['shot_num']
        self.prompt_template_for_prediction = None
        self.prompt_template_for_locate = None
        self.initialize_prompt()
        
        self.hf_modelname_dict = {
            "llama-2-7b"     : "meta-llama/Llama-2-7b-chat-hf",
            "llama-2-13b"    : "meta-llama/Llama-2-13b-chat-hf",
            "llama-2-70b"    : "meta-llama/Llama-2-70b-chat-hf",
            "llama-3-8b"     : "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama-3-70b"    : "meta-llama/Meta-Llama-3-70B-Instruct",
            "llama-3.1-8b"   : "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama-3.1-70b"  : "meta-llama/Meta-Llama-3.1-70₩B-Instruct",
            "llama-3.1-405b" : "meta-llama/Meta-Llama-3.1-405B-Instruct",
        }
        self.model_id = self.hf_modelname_dict.get(params['baseline']['model'].lower(), None)
        self.sanity_check()
        with open(os.path.join('private_key', 'key.json'), 'r') as f:
            private_key = json.load(f)

        self.pipeline = transformers.pipeline(
            "text-generation", model = self.model_id,  model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",
            token = private_key['huggingface_token'],
            max_new_tokens=50,
            temperature=0.9, 
            top_k=25, 
            top_p=0.9,
            return_full_text=False
        )

    def predict(self, pair):        
        prompts = self.finalize_prompt(pair, mode = 'predict')
        print("prompt[0]:")
        print(prompts[0])
        print()
        pred = self.pipeline(prompts)

        for i, p in enumerate(pred):
            print(i)
            print(p)
            print(p[0]['generated_text'])
            print()
        pred = [p[0]['generated_text'] for p in pred]
        pred = self.post_process_prediction(pred)
        return pred
    
    def locate(self, pair):

        raise NotImplementedError


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
        self.prompt_template_for_prediction = "Tell me whether the following question and answer pairs are consistent or inconsistent. \n"
        self.prompt_template_for_locate = "Find the one question and answer pair among the following that is inconsistent with the rest. Respond succinctly, only with the number of the pair.\n"
        
        if self.shot_num == 0:
            return
        
        # (2) few-show examples
        with open(self.few_show_prompts_path, 'r') as f:
            prompt_dict = json.load(f)
        
        prompt_dicts_for_task = prompt_dict[self.params['dataset']]
        prompt_dict_for_prediction= prompt_dicts_for_task['prediction']
        prompt_list_for_prediction = [val for key, val in prompt_dict_for_prediction.items() if int(key) <= self.shot_num]
        if len(prompt_list_for_prediction) < self.shot_num:
            print(f"Number of few_shot examples is less than you want.")
            print(f"Currently we have {len(prompt_list_for_prediction)}, while you want {self.shot_num}.")
            print(f"Nontheless, we don't raise any error. We utilize {len(prompt_list_for_prediction)} number of few-shot examples")
            self.shot_num = len(prompt_list_for_prediction)
            self.params['baseline']['shot_num'] = len(prompt_list_for_prediction)

        for i, prompt in enumerate(prompt_list_for_prediction):
            self.prompt_template_for_prediction += f"\n[example {i+1}]\n{prompt}\n\n"

    def sanity_check(self):
        if self.model_id == None:
            print("======")
            print("[Sanity Check Failed]")
            print(f"Invalid Model Name. Your name is {self.params['baseline']['model']}.")
            print("However, it must be one of:")
            for key, val in self.hf_modelname_dict.items():
                print(key)
                print(f"(which will be converted into {val})")
                print()
            print("======")

    def finalize_prompt(self, pairs, mode):
        prompts = []
        if mode == 'predict':
            front_prompt = self.prompt_template_for_prediction
            end_prompt = "\nYour answer should be  one of 'consistent' and 'inconsistent'. \n Your answer: "

        elif mode == 'locate':
            front_prompt = self.prompt_template_for_locate
            end_prompt = "" # No end prompt here

        else:
            print(f"Invalid mode here. Your mode is {mode}.")
            raise NotImplementedError

        for pair in pairs:
            tmp = front_prompt
            if self.shot_num !=0:
                tmp += "[QA pairs you should consider]\n"
            for i, p in enumerate(pair[0]):
                tmp += f"({i+1}) {p}"
            tmp += " \n "
            tmp += end_prompt
            prompts.append(tmp)

        return prompts
    
    def post_process_prediction(self, prediction):
        # Not completed yet.

        pred = []
        for p in prediction:
            pred.append(p)

        return pred

    def post_process_locate(self, location):
        raise NotImplementedError