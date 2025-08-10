- How to train
    -- 
    * step 1. Determine the arguments

        *  loss_type (e.x., triplet loss. Default = triplet)
        * decomposition (e.x., pairwise, x-y, no. Default = no)
        * representation model (e.x., RoBERTa, Longformer. Default = roberta)
        * task (e.x., multi-hop, vqa)
        * dataset (e.x., musique, lconvqa)
    
    * step 2. Train the model using the arguments in step 1:
        
        * example:
        
            * python train.py --task vqa --dataset lconvqa --loss_type triplet --decomposition no --repre_model roberta
        * If you want to train the model with "WandB", then utilize the "train_wdb.py" instead of "train.py"

    * By default, the pre-trained model will be stored at:
        
        * results/task/dataset/{repre-model}-{decomposition}-{loss_type}

- How to evaluate the pre-trained model
    -- 
    - Step 1. Similar to the training stage, determine the arguments you want.
    - Step 2. Evaluate the model using "evaluate.py":
        
        * example:
            * python evaluate.py --task vqa --dataset lconvqa --loss_type triplet --decomposition no --repre_model roberta

- How to evaluate the baseline model
    -- 
    - Step 1. Similar to the training stage, determine the arguments you want.
        
        - In here, the arguments you should determine are:
        - task
        - dataset
        - type
        - model
        - shot_num
    - Step 2. Evaluate the model using "evaluate_baseline.py":
        
        * example:
            * python evaluate_baseline.py --task vqa --dataset lconvqa --type llm --model llama-2-7b --shot_num 1


- How to run locate
    - Step 1. Make sure you have an energy model already trained.
    - Step 2. Update config.yaml to set the right hyperparameters for locate method.
    - Step 2. Run the following script.
        * example: 
            * python locate.py --task vqa --dataset lconvqa --loss_type triplet --decomposition no --time_key <time-key-of-energy-model>

- Setup
    -- 
    - conda env create -f env.yaml
    - conda activate set_consistency

    - If it fails, then:

        * 1. Install the gpu-version of pytorch (https://pytorch.org/get-started/locally/)
        * 2. pip install -r requirements.txt

- download datasets
    --
    - download the following datasets, and upload at "datasets" folder.
    - L-ConVQA dataset: https://arijitray1993.github.io/ConVQA/Logical_ConVQA.zip    
    - Musique dataset: https://huggingface.co/datasets/voidful/MuSiQue
