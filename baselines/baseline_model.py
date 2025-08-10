from .LLM.LLaMA import LLaMA
from .LLM.GPT import GPT

class baseline_model():
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.baseline_type = None
        self.model_name = None
        self.model = None
        self.initialize()

    def initialize(self):
        self.baseline_type = self.params['baseline']['type']
        
        if self.baseline_type.lower() == 'llm':
            self.model_name = self.params['baseline']['model']
            if 'llama' in self.model_name.lower():
                self.model = LLaMA(self.params)
            elif 'gpt' in self.model_name.lower() or 'deepseek' in self.model_name.lower():
                print("GPT models")
                self.model = GPT(self.params)
            else:
                print(f"Invalid (baseline_type, model_name) pair.\nYou gave (baseline_type, model_name) = ({self.baseline_type}, {self.model_name}).")
                raise NotImplementedError
        
        elif self.baseline_type.lower() == 'nli':
            raise NotImplementedError
        
        else:
            print(f"Invalid baseline_type. You gave '{self.baseline_type}'.")
            raise NotImplementedError
        
    def predict(self, pairs):

        pred = self.model.predict(pairs)

        gold = []
        for p in pairs:
            incon_list = p[1]
            if len(incon_list) == 0:
                gold += [0]
            else:
                gold += [1]

        return {
            "pred": pred,
            "gold": gold,
        }
    
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
        # Obtain LLM inference results separately
        # example: predictions = ["more", "more", "less", "less"]
        predictions = self.model.wiqa_separate_inference(pairs)
        predictions = [p.replace("_", "  ").replace("[", "").replace("]", "") for p in predictions]
        golds = [pairs["qa_pairs"][idx]["answer_label"] for idx in range(len(pairs["qa_pairs"]))]
        answer_set = [pairs["choices"][idx]["text"] for idx in range(len(pairs["choices"])) if "no" not in pairs["choices"][idx]["text"]]

        # Check consistency
        # Note that, the list "predicted_correctly" cannot perfectly check consistency of LLMs. 
        # However, it can be used as a "plausible approximation" for consistency of LLMs.
        predicted_correctly = []
        for idx, answer in enumerate(predictions):
            if answer == golds[idx]:
                predicted_correctly.append(True)    
            else:
                predicted_correctly.append(False)
                if answer not in answer_set:
                    print(f"[warning] Current answer is not in the gold answer set. Responded answer is [{answer}]. But the gold answer set is {answer_set}.")

        if all(predicted_correctly):
            gold_consistency = True
            gold_consistency_by_correct = True
            gold_consistency_by_wrong = False
        elif all([not p for p in predicted_correctly]):
            gold_consistency = True
            gold_consistency_by_correct = False
            gold_consistency_by_wrong = True
        else:
            gold_consistency = False
            gold_consistency_by_correct = False
            gold_consistency_by_wrong = False
        

        return {
            "pred": predictions,
            "gold": golds,
            "consistency": gold_consistency,
            "set_size": len(golds),
            "gold_consistency_by_correct": gold_consistency_by_correct,
            "gold_consistency_by_wrong": gold_consistency_by_wrong,
        }
    
    def wiqa_consistency_check_for_gold(self, pairs):
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
        # Obtain LLM inference results separately
        # example: predictions = ["no effect", "more", "no effect", "no effect"]
        
        prompt = "Tell me whether the following question-answer pairs are consistent or inconsistent. \n"

        qa_pairs = pairs["qa_pairs"]
        for idx, qa_pair in enumerate(qa_pairs):
            prompt += f"{(idx)} question: {qa_pair['question']} answer: {qa_pair['answer_label']}.\n"

        prompt += "Please think step by step: first, clearly articulate your thought process; then, provide your final consistency judgment by choosing either 'consistent' or 'inconsistent' After the 'Consistency:' mark. \n Consistency: "

        result = self.model.api_call(prompt)
        result = self.model.post_process_prediction(result)
        if result == 0:
            return True
        else:
            return False
    
    def wiqa_consistency_check_for_pred(self, pairs, predictions):
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
        # Obtain LLM inference results separately
        # example: predictions = ["no effect", "more", "no effect", "no effect"]
        
        prompt = "Tell me whether the following question-answer pairs are consistent or inconsistent. \n"

        qa_pairs = pairs["qa_pairs"]
        for idx, qa_pair in enumerate(qa_pairs):
            prompt += f"{(idx)} question: {qa_pair['question']} answer: {predictions[idx]}.\n"

        prompt += "Your answer should be only one word, either 'consistent' or 'inconsistent'. \n Consistency: "

        result = self.model.api_call(prompt)
        result = self.model.post_process_prediction(result)
        if result == 0:
            return True
        else:
            return False
        
    def halueval_qa_consistency_verification(self, pair):
        """

        ======== (This function is generated by the assist of GPT-4o) ========

        Evaluate whether the provided answer is factually consistent with the given knowledge.

        :param pair: A dictionary containing:
            - "knowledge": Background information that contains the factual context.
            - "question": The question asked based on the knowledge.
            - "answer": A response to the question, which may be either correct or hallucinated.
        
        :return: A judgment on whether the answer is factually correct or hallucinatory.
        
        Example:
            pair = { 
                "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. First for Women is a woman's magazine published by Bauer Media Group in the USA.",
                "question": "Which magazine was started first Arthur's Magazine or First for Women?", 
                "answer": "First for Women was started first."
            }

        Expected LLM Prompt:
        ---
        Given the following factual knowledge, question, and answer, determine whether the answer is factually accurate based on the knowledge. If the answer contradicts the knowledge or introduces new information not supported by the knowledge, classify it as "Hallucinated". Otherwise, classify it as "Factually Correct".
        
        Knowledge:
        {knowledge}
        
        Question:
        {question}
        
        Answer:
        {answer}
        
        Classification (choose one): 
        - Factually Correct
        - Hallucinated

        Justification:
        Provide a brief explanation of why the answer is classified as "Factually Correct" or "Hallucinated."
        ---
        """

        prompt = f"""
        Given the following factual knowledge, question, and answer, determine whether the answer is factually accurate based on the knowledge. If the answer contradicts the knowledge or introduces new information not supported by the knowledge, classify it as "Hallucinated". Otherwise, classify it as "Factually Correct".
        
        Knowledge:
        {pair['knowledge']}
        
        Question:
        {pair['question']}
        
        Answer:
        {pair['answer']}
        
        Classification (choose one): 
        [Factually Correct]
        [Hallucinated]

        Justification:
        Provide a brief explanation of why the answer is classified as "Factually Correct" or "Hallucinated."
        """
        
        result = self.model.api_call(prompt)
        result = self.halueval_qa_post_process(result)
        if result == 'Factually Correct':
            return 0
        elif result == 'Hallucinated':
            return 1
        else:
            return result
        
    
    def halueval_qa_consistency_verification2(self, pair):
        """

        ======== (This function is generated by the assist of GPT-4o) ========

        Evaluate whether the provided answer is factually consistent with the given knowledge.

        :param pair: A dictionary containing:
            - "knowledge": Background information that contains the factual context.
            - "statement": convert QA pair as a plain sentence

        :return: A judgment on whether the answer is factually correct or hallucinatory.
        

        Expected LLM Prompt:
        ---
        Given the following factual knowledge, determine whether the statment is factually accurate based on the knowledge. If the statment contradicts the knowledge or introduces new information not supported by the knowledge, classify it as "Hallucinated". Otherwise, classify it as "Factually Correct".
        
        Knowledge:
        {pair['knowledge']}
        
        Statment:
        {pair['statement']}
        
        Classification (choose one): 
        [Factually Correct]
        [Hallucinated]

        Justification:
        Provide a brief explanation of why the statement is classified as "Factually Correct" or "Hallucinated."
        ---
        """

        prompt = f"""
        Given the following factual knowledge, determine whether the statment is factually accurate based on the knowledge. If the statment contradicts the knowledge or introduces new information not supported by the knowledge, classify it as "Hallucinated". Otherwise, classify it as "Factually Correct".
        
        Knowledge:
        {pair['knowledge']}
        
        Statment:
        {pair['statement']}
        
        Classification (choose one): 
        [Factually Correct]
        [Hallucinated]

        Justification:
        Provide a brief explanation of why the statement is classified as "Factually Correct" or "Hallucinated."
        """
        
        result = self.model.api_call(prompt)
        result = self.halueval_qa_post_process(result)
        if result == 'Factually Correct':
            return 0
        elif result == 'Hallucinated':
            return 1
        else:
            return result
        

    def halueval_qa_post_process(self, result):
        
        """not implemented yet"""
        # print("prompt:", result)
        if "[Factually Correct]" in result:
            return "Factually Correct"
        if "[Hallucinated]" in result:
            return "Hallucinated"


    def halueval_qa_data_modification(self, pair):
        """
        Convert a question-answer pair into a natural sentence.

        :param pair: A dictionary containing:
            - "question": The question asked.
            - "answer": The corresponding response (which may be factually correct or hallucinated).
        
        :return: A transformed sentence that conveys the same meaning as the QA pair in a natural language form.
        
        Example:
            pair = { 
                "question": "Which magazine was started first Arthur's Magazine or First for Women?", 
                "answer": "First for Women was started first."
            }

        Expected LLM Prompt:
        ---
        Given the following question and answer pair, rewrite them as a natural, well-structured sentence while preserving the original meaning. Ensure that the sentence is grammatically correct, fluent, and maintains factual consistency with the given knowledge.
        
        Question:
        {question}
        
        Answer:
        {answer}
        
        Transformed Sentence:
        [Output a single sentence that expresses the same meaning as the question-answer pair]
        ---
        """

        prompt = f"""
        Given the following question and answer pair, rewrite them as a natural, well-structured sentence while preserving the original meaning. Ensure that the sentence is grammatically correct, fluent, and maintains factual consistency with the given knowledge.
    
        
        Question:
        {pair['question']}
        
        Answer:
        {pair['answer']}
        
        Transformed Sentence:
        [Output a single sentence that expresses the same meaning as the question-answer pair]
        """

        result = self.model.api_call(prompt)
        return result




    def locate(self, pair):
        # assert len(pair) == 1

        gold = pair[0][1]               # list of true inconsistent pairs
        gold = [g+1 for g in gold]
        pred = self.model.locate(pair)  # list of predicted inconsistent pairs
        pair_num = len(pair[0][0])
        if set(pred) == set(gold):
            correct =1
        else:
            correct = 0

        # precision
        if len(pred) == 0:
            precision = 1
        else:
            precision = len(set(pred) & set(gold)) / len(set(pred))

        # recall
        if len(gold) == 0:
            recall = 1
        else:
            recall = len(set(pred) & set(gold)) / len(set(gold)) 

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)
        accuracy = correct
            
        return {
            "pair_num": pair_num,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }