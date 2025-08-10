import os, json, pickle, itertools
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data import Dataset

class l_convqa_dataset(Dataset):
    def __init__(self, params, mode = 'train', consistency = 'con', pairwise = 'arbitrary_pairs'):
        super().__init__()
        self.params = params
        self.pairwise = pairwise
        self.seed = params['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.mode = mode # one of 'train', 'eval', 'eval2', 'test.
        self.consistency = consistency
        self.folder_path = os.path.join("datasets",
                        "convqa", "dataset_MTURK_refine")
        self.dataset = None
        if pairwise == 'two_pairs':
            self.load_dataset()
        elif pairwise == 'arbitrary_pairs':
            self.load_dataset2()
        else:
            print(f"Invalid pairwise model. Current params['pairwise'] seems to be {self.params['pairwise']}")

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def load_dataset(self):

        consistent_data = []
        inconsistent_data = []
        files = os.listdir(self.folder_path)
        for file in files:
            if not file.endswith("json"):
                continue
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, 'r') as f:
                one_json = json.load(f)
                con = one_json['consistent']
                incon = one_json['inconsistent']
                min_len  = min(len(con), len(incon))

                for idx, c in enumerate(con):
                    if idx >= min_len:
                        break
                    tmp = []
                    random.shuffle(c)
                    for cc in c:
                        tmp.append((cc['question'], cc['answer'], True))
                    consistent_data.append(tmp)
                for idx, ic in enumerate(incon):
                    if idx >= min_len:
                        break
                    tmp = []
                    for icc_idx, icc in enumerate(ic):
                        if icc_idx==0:
                            tmp.append((icc['question'], icc['answer'], True))
                        elif icc_idx == 1:
                            tmp.append((icc['question'], icc['answer'], False))

                    inconsistent_data.append(tmp)

        # train : dev : test = 6 : 2 : 2
        random.shuffle(consistent_data)
        random.shuffle(inconsistent_data)

        if self.mode == 'train':
            con_data = consistent_data[0:int(len(consistent_data) * 0.6)]
            incon_data = inconsistent_data[0:int(len(inconsistent_data) * 0.6)]
        elif self.mode in {'dev', 'eval'}:
            con_data = consistent_data[int(len(consistent_data) * 0.6):int(len(consistent_data) * 0.8)]
            incon_data = inconsistent_data[int(len(inconsistent_data) * 0.6):int(len(inconsistent_data) * 0.8)]
        elif self.mode == 'test':
            con_data = consistent_data[int(len(consistent_data) * 0.8):]
            incon_data = inconsistent_data[int(len(inconsistent_data) * 0.8):]
        else:
            raise f"Invalid mode = {self.mode}. \nMust be one of 'train', 'dev'(or 'eval'), 'test'"

        if self.consistency in {'con', 'consistent', 'consistency'}:
            data = con_data
        elif self.consistency in {'incon', 'inconsistent', 'inconsistency'}:
            data = incon_data

        self.dataset = data

    def dataset_check(self):
        consistent_data = []
        inconsistent_data = []
        files = os.listdir(self.folder_path)

        error_file = set([])
        for file in tqdm(files):
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, 'r') as f:
                one_json = json.load(f) # e.g., qas_1.json
            con = one_json['consistent']
            incon = one_json['inconsistent']
            
            # 비어있는 파일 건너뛰기
            if len(con) == 0:
                continue 

            # 확인해보고 싶은 사항들.
            # Assumption 1. 하나의 con-set에 등장하는 모든 x에 대하여, incon-set을 샅샅히 뒤져보면 그와 동일한 x가 반드시 존재할 것이다.
                # 이때, x는 어떠한 적어도 하나의 incon-set의 pair들 중 최소 2번째 이상의 자리여야 한다.
            # Assumption 2. assumption 1과 같이 con/incon에서 찾은 서로 같은 x들과 그들의 y값의 pairs (x, y_con), (x, y_incon)에 대하여,
                # y_con != y_incon일 것이다.
                # 특히, y_con=="yes"라면 y_incon=="no"일 것이고, 비슷하게 y_con=="no"라면 y_incon=="yes"일 것이다.

            # 이러면 결론내릴 수 있는 점:
                # in-con set을 우리가 완전히 통제 가능한 형태로 만들 수 있다.
                # 구체적으로, in-con set에서 어느 부분이 잘못된 pair인지를 짚어줄 수 있다. 
                # 따라서 locate and edit을 평가하는 것이 가능해진다.

            # 실제 확인 결과
                # Assumption 1이 성립하지는 않았음. 대신 아래가 성립하므로, 실험은 가능:
                    # "하나의 con-set에 등장하는 모든 x에 대하여" 대신,
                    # "하나의 con-set에 등장하는 x들 중 적어도 하나에 대하여"가 성립함.


            for c in con:
                con_qa_pairs = [(cc['question'], cc['answer']) for cc in c]
                con_qs = [con_qa_pair[0] for con_qa_pair in con_qa_pairs]
                con_as = [con_qa_pair[1] for con_qa_pair in con_qa_pairs]
                find_all_same_q_set = False
                and_answer_different = False

                for idx, ic in enumerate(incon):
                    ic_qa_pairs = [(icc['question'], icc['answer']) for icc in ic]
                    incon_qs = [ic_qa_pair[0] for ic_qa_pair in ic_qa_pairs]
                    incon_as = [ic_qa_pair[1] for ic_qa_pair in ic_qa_pairs]

                    if all([ic_q in  con_qs for ic_q in incon_qs]):
                        find_all_same_q_set = True

                        for con_idx, (con_q, con_a) in enumerate(con_qa_pairs):
                            if incon_qs[1] == con_q and incon_as[1] != con_a:
                                and_answer_different = True
                    
                    if find_all_same_q_set and and_answer_different:
                        break
                if find_all_same_q_set and and_answer_different:
                    break

            if find_all_same_q_set == False:
                print("==============")
                print(f"error. find_all_same_q_set is False. {file_path}")
                error_file.add(file_path)
            if and_answer_different == False: 
                print(f"error. and_answer_different is False. {file_path}")

        print("error files:")
        for e in error_file:
            print(e)

        print("ended")
            # for idx, c in enumerate(con):
            #     find_same_q = False
            #     answer_different = False
            #     yes_no_reversed = True
            #     for cc in c:
            #         # consistent set에 있는 하나의 (question, answer) pair.
            #         q = cc['question']
            #         a = cc['answer']
            #         for in_idx, ic in enumerate(incon):
            #             qa_incon = [(icc['question'], icc['answer']) for icc in ic]
            #             for (q_incon, a_incon) in qa_incon[1:]:
            #                 if (q_incon == q):
            #                     find_same_q = True
            #                     if (a.lower() == 'yes' and a_incon.lower() == 'no'):
            #                         answer_different = True
            #                         yes_no_reversed = True
            #                         break
            #                     elif (a.lower() == 'yes' and a_incon.lower() != 'no'):
            #                         answer_different = True
            #                         yes_no_reversed = False
            #                     elif (a.lower() == 'no' and a_incon.lower() == 'yes'):
            #                         answer_different = True
            #                         yes_no_reversed = True
            #                         break
            #                     elif (a.lower() == 'no' and a_incon.lower() != 'yes'):
            #                         answer_different = True
            #                         yes_no_reversed = False
                                    
            #                     elif (a.lower() != a_incon.lower()):
            #                         answer_different = True
            #                         break
            #     if find_same_q == False:
            #         print(f"we couldn't found same question for '{q}'. idx={idx}, {file_path}")
            #     if answer_different == False:
            #         print(f"we couldn't found same answer & different answer pair for ('{q}', '{a}'). idx={idx}, {file_path}")
    
            #     # print()
                                    


    def load_dataset2(self):

        consistent_data = []
        inconsistent_data = []
        files = os.listdir(self.folder_path)

        for file in files:
            if not file.endswith("json"):
                continue
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, 'r') as f:
                one_json = json.load(f)
                con = one_json['consistent']
                incon = one_json['inconsistent']
                # min_len  = min(len(con), len(incon))

                for idx, c in enumerate(con):

                    # for consistency data
                    tmp_con = []
                    for cc in c:
                        tmp_con.append((cc['question'], cc['answer'], True))
                    consistent_data.append(tmp_con)

                    # for inconsistency data
                    tmp_incon = [t for t in tmp_con]
                    q_incon, a_incon, idx_incon = self.detect_inconsistent_data(
                        tmp_con, incon
                    )

                    tmp_incon[idx_incon] = (q_incon, a_incon, False)
                    inconsistent_data.append(tmp_incon)

        # train : dev : test = 6 : 2 : 2
        random.shuffle(consistent_data)
        random.shuffle(inconsistent_data)

        if self.mode == 'train':
            con_data = consistent_data[0:int(len(consistent_data) * 0.6)]
            incon_data = inconsistent_data[0:int(len(inconsistent_data) * 0.6)]
        elif self.mode in {'dev', 'eval'}:
            con_data = consistent_data[int(len(consistent_data) * 0.6):int(len(consistent_data) * 0.8)]
            incon_data = inconsistent_data[int(len(inconsistent_data) * 0.6):int(len(inconsistent_data) * 0.8)]
        elif self.mode == 'test':
            con_data = consistent_data[int(len(consistent_data) * 0.8):]
            incon_data = inconsistent_data[int(len(inconsistent_data) * 0.8):]
        else:
            raise f"Invalid mode = {self.mode}. \nMust be one of 'train', 'dev'(or 'eval'), 'test'"

        if self.consistency in {'con', 'consistent', 'consistency'}:
            data = con_data
        elif self.consistency in {'incon', 'inconsistent', 'inconsistency'}:
            data = incon_data

        self.dataset = data
        print(f"load_dataset completed. dataset length: {len(self.dataset)}")

    def detect_inconsistent_data(self, tmp_con, incon):
        indices = [i for i in range(len(tmp_con))]
        random.shuffle(indices)

        for i in indices:
            (q, a, _) = tmp_con[i]
            random.shuffle(incon)
            for idx, ic in enumerate(incon):
                qa_incon = [(icc['question'], icc['answer']) for icc in ic][1:]
                for (q_incon, a_incon) in qa_incon:
                    if (q == q_incon) and (a != a_incon):
                        return q_incon, a_incon, i

        return None
    


class l_convqa_fine_grained_dataset(Dataset):
    def __init__(self, params, mode = 'train', 
                 semantic = 'con', 
                 pairwise = 'arbitrary_pairs', 
                 load_in_once_everything = False, 
                 output_form = 'real_num', # if output_form == 'real_num', then the data is loaded for energy netowrk. elif output_form == '2dim_vec', then the data is loaded for the binary class model (supervised loss)
                 input_type = "set"
                 ): 
        super().__init__()
        self.params = params
        self.pairwise = pairwise
        self.seed = params['seed']
        self.output_form = output_form
        self.input_type = input_type
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.mode = mode # one of 'train', 'eval', 'eval2', 'test.
        self.dataset_num = {"train":self.params['lconvqa']['stepwise_dataset_train_num'], 
                            "eval":self.params['lconvqa']['stepwise_dataset_eval_num'], 
                            "eval2":self.params['lconvqa']['stepwise_dataset_eval2_num'], 
                            "test":self.params['lconvqa']['stepwise_dataset_test_num']}[self.mode]
        self.load_everything = load_in_once_everything # T, F, 'divide_by_semantic', 'divide_by_length'
        if self.mode == 'dev':
            self.mode == 'eval'
        self.semantic = semantic # one of 'con', 'incon', 'con_and_incon', 'incon_and_incon'
        self.folder_path = os.path.join("datasets",
                        "convqa", "dataset_MTURK_refine") # 다운로드받은 raw dataset을 모아두는 폴더.
        self.processed_dataset_path = os.path.join("datasets",
                        "convqa", 'processed_data')
        self.dataset = None
        self.load_dataset()
        random.Random(self.seed).shuffle(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

    def load_dataset(self):

        # print(f"dataset loaded: {self.mode} - {self.pairwise} - {self.semantic}.")
        required_raw_dataset = [f"raw_{mode}_filenames.pickle" for mode in ['train', 'eval', 'eval2', 'test']]
        if not all([r in os.listdir(self.processed_dataset_path) for r in required_raw_dataset]):
            self.raw_data_split()

        self.dataset = []
        if self.output_form == '2dim_vec':
            if self.input_type == 'set':
                if self.load_everything == True and type(self.load_everything) == bool:
                    # self.dataset += self.load_dataset_two_pairs_con()
                    # self.dataset += self.load_dataset_two_pairs_incon()
                    # self.dataset += self.load_dataset_two_pairs_con_and_incon()
                    # self.dataset += self.load_dataset_two_pairs_incon_and_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()
                elif self.load_everything == "only_con_and_incon":
                    self.dataset = []
                    # self.dataset += self.load_dataset_two_pairs_con()
                    # self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_arbitrary_pairs_con()
                    self.dataset += self.load_dataset_arbitrary_pairs_incon()
            elif self.input_type.lower() in {'oto', 'one_to_one'}:
                transformed_dataset = transform_arbitrary_pairs_to_two_pairs(
                    [
                        self.load_dataset_two_pairs_con(),
                        self.load_dataset_two_pairs_incon()
                    ]
                )
                self.dataset +=transformed_dataset[0]
                self.dataset +=transformed_dataset[1]
        
        if self.output_form == 'real_num':
            if self.input_type == 'set':
                if self.load_everything == True and type(self.load_everything) == bool:
                    self.dataset = []
                    if self.semantic == 'con':
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_con()

                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()

                    elif self.semantic == 'incon':
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()

                        self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()

                else:
                    if self.load_everything == 'divide_by_semantic':
                        self.dataset = []
                        if self.semantic == 'con':
                            # self.dataset += self.load_dataset_two_pairs_con()
                            self.dataset += self.load_dataset_arbitrary_pairs_con()
                        elif self.semantic == 'incon':
                            # self.dataset += self.load_dataset_two_pairs_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon()
                        elif self.semantic == 'con_and_incon':
                            # self.dataset += self.load_dataset_two_pairs_con_and_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                        elif self.semantic == 'incon_and_incon':
                            # self.dataset += self.load_dataset_two_pairs_incon_and_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()
                    elif self.load_everything == 'divide_by_length':
                        self.dataset = []
                        if self.pairwise == 'two_pairs':
                            self.dataset += self.load_dataset_two_pairs_con()
                            self.dataset += self.load_dataset_two_pairs_incon()
                            self.dataset += self.load_dataset_two_pairs_con_and_incon()
                            self.dataset += self.load_dataset_two_pairs_incon_and_incon()
                        elif self.pairwise == 'arbitrary_pairs':
                            self.dataset += self.load_dataset_arbitrary_pairs_con()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                            self.dataset += self.load_dataset_arbitrary_pairs_incon_and_incon()
                    else:
                        if self.pairwise == 'two_pairs':
                            if self.semantic == 'con':
                                self.dataset = self.load_dataset_two_pairs_con()
                            elif self.semantic == 'incon':
                                self.dataset = self.load_dataset_two_pairs_incon()
                            elif self.semantic == 'con_and_incon':
                                self.dataset = self.load_dataset_two_pairs_con_and_incon()
                            elif self.semantic == 'incon_and_incon':
                                self.dataset = self.load_dataset_two_pairs_incon_and_incon()
                            else:
                                raise f"invalid dataset mode: {self.pairwise} - {self.semantic}"
                        elif self.pairwise == 'arbitrary_pairs':
                            if self.semantic == 'con':
                                self.dataset = self.load_dataset_arbitrary_pairs_con()
                            elif self.semantic == 'incon':
                                self.dataset = self.load_dataset_arbitrary_pairs_incon()
                            elif self.semantic == 'con_and_incon':
                                self.dataset = self.load_dataset_arbitrary_pairs_con_and_incon()
                            elif self.semantic == 'incon_and_incon':
                                self.dataset = self.load_dataset_arbitrary_pairs_incon_and_incon()
                            else:
                                raise f"invalid dataset mode: {self.pairwise} - {self.semantic}"
            
            elif self.input_type.lower() in {'oto', 'one_to_one'}:
                if self.semantic == 'con':
                    self.dataset += self.load_dataset_two_pairs_con()
                    self.dataset += self.load_dataset_two_pairs_con()
                    self.dataset += self.load_dataset_two_pairs_con()

                    # self.dataset += self.load_dataset_arbitrary_pairs_con_and_incon()
                    # self.dataset += self.load_dataset_arbitrary_pairs_incon()

                elif self.semantic == 'incon':
                    self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_two_pairs_incon()
                    self.dataset += self.load_dataset_two_pairs_incon()

                    

        print(f"dataset loaded: {self.mode} - {self.pairwise} - {self.semantic}. With length {len(self.dataset)}")

    def load_dataset_two_pairs_con(self):
        dataset_name = f"two_pairs_con_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list = pickle.load(f)

        
        
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                one_json = json.load(f)
            con = one_json['consistent']
            incon = one_json['inconsistent']
            
            for idx, ic in enumerate(incon):
                c = self.generate_two_pair_con_data_wrt_incon(con, ic)
                if c == None:
                    continue
                data.append(c)
        
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset
            
    def load_dataset_two_pairs_incon(self):
        dataset_name = f"two_pairs_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list = pickle.load(f)
        
        
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                one_json = json.load(f)
            con = one_json['consistent']
            incon = one_json['inconsistent']
            
            for idx, ic in enumerate(incon):
                tmp = []
                for icc_idx, icc in enumerate(ic):
                    if icc_idx==0:
                        tmp.append((icc['question'], icc['answer'], True))
                    elif icc_idx == 1:
                        tmp.append((icc['question'], icc['answer'], False))
                random.shuffle(tmp)
                data.append(tmp)
        
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)    
        return dataset

    def load_dataset_two_pairs_con_and_incon(self):
        dataset_name = f"two_pairs_con_and_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list1 = pickle.load(f)
        
        
        file_path_list2 = file_path_list1[:]
        random.Random(self.seed).shuffle(file_path_list2)

        count = 0
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path1, file_path2 in zip(file_path_list1, file_path_list2):
            with open(file_path1, 'r') as f:
                one_json1 = json.load(f)
            con1 = one_json1['consistent']
            incon1 = one_json1['inconsistent']

            with open(file_path2, 'r') as f:
                one_json2 = json.load(f)
            con2 = one_json2['consistent']
            incon2 = one_json2['inconsistent']

            incon_minlen = min(len(incon1), len(incon2))
            
            for idx in range(incon_minlen):
                ic1 = incon1[idx]
                ic2 = incon2[idx]
                tmp = []

                if count%2 == 0:
                    for icc_idx, icc in enumerate(ic1):
                        if icc_idx==0:
                            tmp.append((icc['question'], icc['answer'], True))
                        elif icc_idx == 1:
                            tmp.append((icc['question'], icc['answer'], False))
                
                    c = self.generate_two_pair_con_data_wrt_incon(con2, ic2)
                    if c == None:
                        continue
                    tmp += c
                elif count%2 == 1:
                    for icc_idx, icc in enumerate(ic2):
                        if icc_idx==0:
                            tmp.append((icc['question'], icc['answer'], True))
                        elif icc_idx == 1:
                            tmp.append((icc['question'], icc['answer'], False))
                
                    c = self.generate_two_pair_con_data_wrt_incon(con1, ic1)
                    if c == None:
                        continue
                    tmp += c
                random.shuffle(tmp)
                data.append(tmp)
        
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_two_pairs_incon_and_incon(self):
        dataset_name = f"two_pairs_incon_and_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list1 = pickle.load(f)
        
    
        file_path_list2 = file_path_list1[:]
        random.Random(self.seed).shuffle(file_path_list2)

        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path1, file_path2 in zip(file_path_list1, file_path_list2):
            with open(file_path1, 'r') as f:
                one_json1 = json.load(f)
            con1 = one_json1['consistent']
            incon1 = one_json1['inconsistent']

            with open(file_path2, 'r') as f:
                one_json2 = json.load(f)
            con2 = one_json2['consistent']
            incon2 = one_json2['inconsistent']

            incon_minlen = min(len(incon1), len(incon2))
            
            for idx in range(incon_minlen):
                ic1 = incon1[idx]
                ic2 = incon2[idx]
                tmp = []
                for icc_idx, icc in enumerate(ic1):
                    if icc_idx==0:
                        tmp.append((icc['question'], icc['answer'], True))
                    elif icc_idx == 1:
                        tmp.append((icc['question'], icc['answer'], False))
                for icc_idx, icc in enumerate(ic2):
                    if icc_idx==0:
                        tmp.append((icc['question'], icc['answer'], True))
                    elif icc_idx == 1:
                        tmp.append((icc['question'], icc['answer'], False))
            
                random.shuffle(tmp)
                data.append(tmp)
        
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset
    def load_dataset_arbitrary_pairs_con(self):
        dataset_name = f"arbitrary_pairs_con_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list = pickle.load(f)
        
        
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                one_json = json.load(f)
            con = one_json['consistent']
            incon = one_json['inconsistent']
            
            for c in con:
                tmp = []
                for cc in c:
                    tmp.append((cc['question'], cc['answer'], True))
                random.shuffle(tmp)
                data.append(tmp)
        
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset
            
    def load_dataset_arbitrary_pairs_incon(self):
        dataset_name = f"arbitrary_pairs_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list = pickle.load(f)
        
        
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                one_json = json.load(f)
            con = one_json['consistent']
            incon = one_json['inconsistent']
            
            for c in con:
                incon_qa_pair = self.generate_arbitrary_pair_incon_data_wrt_con(incon, c)
                if incon_qa_pair == None:
                    continue
                random.shuffle(incon_qa_pair)
                data.append(incon_qa_pair)

        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_con_and_con(self):
        dataset_name = f"arbitrary_pairs_con_and_con_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list1 = pickle.load(f)
        
        
        file_path_list2 = file_path_list1[:]
        random.Random(self.seed).shuffle(file_path_list2)

        count = 0
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path1, file_path2 in zip(file_path_list1, file_path_list2):
            with open(file_path1, 'r') as f:
                one_json1 = json.load(f)
            con1 = one_json1['consistent']
            incon1 = one_json1['inconsistent']

            with open(file_path2, 'r') as f:
                one_json2 = json.load(f)
            con2 = one_json2['consistent']
            incon2 = one_json2['inconsistent']

            con_minlen = min(len(con1), len(con2))
            for idx in range(con_minlen):
                c1 = con1[idx]
                c2 = con2[idx]
                
                tmp = []
                if count%2 == 0:
                    for cc1 in c1:
                        tmp.append((cc1['question'], cc1['answer'], True))
                    
                    for cc2 in c2:
                        tmp.append((cc2['question'], cc2['answer'], True))
                    
                
                random.shuffle(tmp)
                data.append(tmp)

                
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset
    
    def load_dataset_arbitrary_pairs_con_and_incon(self):
        dataset_name = f"arbitrary_pairs_con_and_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list1 = pickle.load(f)
        
        
        file_path_list2 = file_path_list1[:]
        random.Random(self.seed).shuffle(file_path_list2)

        count = 0
        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path1, file_path2 in zip(file_path_list1, file_path_list2):
            with open(file_path1, 'r') as f:
                one_json1 = json.load(f)
            con1 = one_json1['consistent']
            incon1 = one_json1['inconsistent']

            with open(file_path2, 'r') as f:
                one_json2 = json.load(f)
            con2 = one_json2['consistent']
            incon2 = one_json2['inconsistent']

            con_minlen = min(len(con1), len(con2))
            for idx in range(con_minlen):
                c1 = con1[idx]
                c2 = con2[idx]
                
                tmp = []
                if count%2 == 0:
                    for cc1 in c1:
                        tmp.append((cc1['question'], cc1['answer'], True))
                    
                    ic2 = self.generate_arbitrary_pair_incon_data_wrt_con(incon2, c2)
                    if ic2 == None:
                        continue
                    tmp += ic2
                elif count%2 == 1:
                    for cc2 in c2:
                        tmp.append((cc2['question'], cc2['answer'], True))
                    
                    ic1 = self.generate_arbitrary_pair_incon_data_wrt_con(incon1, c1)
                    if ic1 == None:
                        continue
                    tmp += ic1
                
                random.shuffle(tmp)
                data.append(tmp)
            
            count +=1
            count = count % 2
                
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def load_dataset_arbitrary_pairs_incon_and_incon(self):
        dataset_name = f"arbitrary_pairs_incon_and_incon_{self.mode}_{str(self.dataset_num)}"
        dataset_list = os.listdir(self.processed_dataset_path)
        if dataset_name + ".pickle" in dataset_list:
            with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        
        data = []
        # 데이터 생성에 사용할 raw 데이터 불러오기. train/eval/test에 따라 불러와야 하는 파일 인덱스가 다름.
        with open(os.path.join(self.processed_dataset_path, f"raw_{self.mode}_filenames.pickle"), 'rb') as f:
            file_path_list1 = pickle.load(f)
        
        
        file_path_list2 = file_path_list1[:]
        random.Random(self.seed).shuffle(file_path_list2)

        # 불러온 raw 데이터 사용하여 데이터셋 preprocess
        for file_path1, file_path2 in zip(file_path_list1, file_path_list2):
            with open(file_path1, 'r') as f:
                one_json1 = json.load(f)
            con1 = one_json1['consistent']
            incon1 = one_json1['inconsistent']

            with open(file_path2, 'r') as f:
                one_json2 = json.load(f)
            con2 = one_json2['consistent']
            incon2 = one_json2['inconsistent']

            con_minlen = min(len(con1), len(con2))
            for idx in range(con_minlen):
                c1 = con1[idx]
                c2 = con2[idx]
                
                tmp = []
                ic1 = self.generate_arbitrary_pair_incon_data_wrt_con(incon1, c1)
                ic2 = self.generate_arbitrary_pair_incon_data_wrt_con(incon2, c2)
                if ic1 == None or ic2 == None:
                    continue
                tmp += ic1
                tmp += ic2
                random.shuffle(tmp)
                data.append(tmp)
                
        
        dataset = data[:self.params['lconvqa'][f"stepwise_dataset_{self.mode}_num"]]
        with open(os.path.join(self.processed_dataset_path, f"{dataset_name}.pickle"), 'wb') as f:
            pickle.dump(dataset, f, protocol=1)
        return dataset

    def raw_data_split(self):
        """
        다운로드받은 원본 데이터셋에 대하여 train/eval/test 데이터셋 분리를 진행.
        한 번 분리해두고 나서 그 결과(파일 이름들을 모아둔 리스트)를 파일로 저장. 다음번에 코드를 실행할 때는 그냥 읽기만 하면 되도록.

        다만 raw data가 비어있는 json 파일인 경우도 많아서, 이걸 걸러내는 과정을 추가하였음.
        """
        print("=================")
        print("raw data split start.")
        # 비어있지 않은 json 파일 걸러내기
        nonempty_filenames = []
        files = os.listdir(self.folder_path)
        for file in files:
            if not file.endswith("json"):
                continue
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, 'r') as f:
                one_json = json.load(f)
            if len(one_json['consistent'])!=0 and len(one_json['inconsistent'])!=0:
                nonempty_filenames.append(file_path)

        print("dataset_seed num:", len(nonempty_filenames))

        # random suffle 하고 split
        random.shuffle(nonempty_filenames)
        train_filenames = nonempty_filenames[:int(len(nonempty_filenames) * 0.6)]
        eval_filenames = nonempty_filenames[int(len(nonempty_filenames) * 0.6):int(len(nonempty_filenames) * 0.8)]
        eval2_filenames = nonempty_filenames[int(len(nonempty_filenames) * 0.8):int(len(nonempty_filenames) * 0.9)]
        test_filenames = nonempty_filenames[int(len(nonempty_filenames) * 0.9):]

        # 결과를 저장
        with open(os.path.join(self.processed_dataset_path, 'raw_train_filenames.pickle'), 'wb') as f:
            pickle.dump(train_filenames, f)
        with open(os.path.join(self.processed_dataset_path, 'raw_eval_filenames.pickle'), 'wb') as f:
            pickle.dump(eval_filenames, f)
        with open(os.path.join(self.processed_dataset_path, 'raw_eval2_filenames.pickle'), 'wb') as f:
            pickle.dump(eval2_filenames, f)
        with open(os.path.join(self.processed_dataset_path, 'raw_test_filenames.pickle'), 'wb') as f:
            pickle.dump(test_filenames, f)

        print("raw data split ends.")
        print(f"raw_train_filename: {os.path.join(self.processed_dataset_path, 'raw_train_filenames.pickle')}")
        print(f"raw_train dataset length: {len(train_filenames)}")
        print(f"raw_eval_filename: {os.path.join(self.processed_dataset_path, 'raw_eval_filenames.pickle')}")
        print(f"raw_eval dataset length: {len(eval_filenames)}")
        print(f"raw_eval2_filename: {os.path.join(self.processed_dataset_path, 'raw_eval2_filenames.pickle')}")
        print(f"raw_eval2 dataset length: {len(eval2_filenames)}")
        print(f"raw_test_filename: {os.path.join(self.processed_dataset_path, 'raw_test_filenames.pickle')}")
        print(f"raw_test dataset length: {len(test_filenames)}")
        print("=================")


    def detect_inconsistent_data(self, tmp_con, incon):
        indices = [i for i in range(len(tmp_con))]
        random.shuffle(indices)

        for i in indices:
            (q, a, _) = tmp_con[i]
            random.shuffle(incon)
            for idx, ic in enumerate(incon):
                qa_incon = [(icc['question'], icc['answer']) for icc in ic][1:]
                for (q_incon, a_incon) in qa_incon:
                    if (q == q_incon) and (a != a_incon):
                        return q_incon, a_incon, i

        return None
    
    def generate_two_pair_con_data_wrt_incon(self, con, ic):
        incon_qa_pairs = []
        for icc_idx, icc in enumerate(ic):
            if icc_idx==0:
                incon_qa_pairs.append((icc['question'], icc['answer'], True))
            elif icc_idx == 1:
                incon_qa_pairs.append((icc['question'], icc['answer'], False))

        
        target_q = incon_qa_pairs[-1][0]
        target_q_found = False
        for idx, c in enumerate(con):
            # random.shuffle(c)
            for cc in c:
                if cc['question'] == target_q:
                    con_q, con_a = cc['question'], cc['answer']
                    target_q_found = True
                    break
            if target_q_found:
                break
        if target_q_found == False:
            return None
        
        con_qa_pairs = incon_qa_pairs[:]
        con_qa_pairs[-1] = (con_q, con_a, True)
        return con_qa_pairs
    
    def generate_arbitrary_pair_incon_data_wrt_con(self, incon, c):
        con_qa_pairs = []
        for cc_idx, cc in enumerate(c):
            con_qa_pairs.append((cc['question'], cc['answer'], True))

        incon_a = None
        for idx, (con_q, con_a, _) in enumerate(con_qa_pairs):
            target_q = con_q
            target_q_found = False
            for ic in incon:
                for icc in ic:
                    if icc['question'] == target_q and icc['answer'] != con_a:
                        target_q_found = True
                        incon_a = icc['answer']
                        break
                if target_q_found:
                    break
            if target_q_found:
                break
        if target_q_found == False:
            return None
        
        incon_qa_pairs = con_qa_pairs[:]
        incon_qa_pairs[idx] = (con_q, incon_a, False)

        return incon_qa_pairs
    

def transform_arbitrary_pairs_to_two_pairs(seq_of_dataset):
    """

    seq_of_dataset = dataset들을 모아둔 것. set, tuple, list 모두 관계 없이 파이썬 상에서 iterable하기만 하면 됨.

    dataset: l_convqa_fine_grained_dataset.dataset
    
    
    for d in dataset: # type(dataset) == list
        for data_point in d: # type(d) = list
            question, answer, consistency = data_point # type(data_point) = tuple
            # type(question): usually string
            # type(answer): usually string
            # type(consistency): bool. Consistency indicates whether this qa pair is consistent or inconsistent with respect to the rest.
    
    """

    transformed_dataset_con = []
    transformed_dataset_incon = []
    set_sizes = {}

    for dataset in seq_of_dataset:
        for data in dataset:
            if len(data) == 2:
                if all([d[2] for d in data]):
                    transformed_dataset_con.append(data)
                else:
                    transformed_dataset_incon.append(data)

            else:
                combinations = itertools.combinations(data, 2) # "len(d)-choose-2"-number of combinations
                for two_pair in combinations:
                    if all([p[2] for p in two_pair]):
                        transformed_dataset_con.append(two_pair)
                    else:
                        transformed_dataset_incon.append(two_pair)
            
            if len(data) not in set_sizes:
                set_sizes[len(data)] = 0
            set_sizes[len(data)] +=1


    additional_info = {}
    for i, s in enumerate(seq_of_dataset):
        additional_info[f"original_dataset_number_{i+1}_length"] = len(s)
    additional_info["con_transformed"] = len(transformed_dataset_con)
    additional_info["incon_transformed"] = len(transformed_dataset_incon)
    for i in set_sizes.keys():
        additional_info[f"original_data_set_size_with_|set|={i}"] = set_sizes[i]

    print("===========================")
    print("transformed arbitrary-pairs dataset to two-pairs dataset.")
    for key, val in additional_info.items():
        print('key:', key)
        print('key:', val)
        print("\n")
    print("===========================")


    return transformed_dataset_con, transformed_dataset_incon, additional_info