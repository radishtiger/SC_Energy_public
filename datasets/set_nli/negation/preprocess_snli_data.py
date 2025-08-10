import pandas as pd
import os
import openai
import json
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")


def ask(instruction, text):
    messages = [{"role": "system", "content": instruction}]
    user_input = {"role": "user", "content":text}
    messages.append(user_input)

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages
    )

    bot_text = response.choices[0].message.content
    return bot_text

def get_negation(text):
    instruction_negation =f"""
    Premise: An older man wearing a salon drape getting a haircut.
    Hypothesis: A man gets a haircut.
    Negation of Hypothesis: No man gets a haircut.
    Negation of Premise: No older man wearing a salon drape is getting a haircut.
    Premise: A man with glasses sitting at a restaurant staring at something that is not shown.
    Hypothesis: A man at a restaurant.
    Negation of Hypothesis: No man at a restaurant.
    Negation of Premise: No man with glasses sitting at a restaurant staring at something that is not shown.
    Premise: The school is having a special event in order to show the American culture on how other cultures are dealt with in parties.
    Hypothesis: A school is hosting an event.
    Negation of Hypothesis: No school is not hosting an event.
    Negation of Premise: The school is not having a special event to show the American culture or how other cultures are dealt with in parties.
    Premise: High fashion ladies wait outside a tram beside a crowd of people in the city.
    Hypothesis: Women are waiting by a tram.
    Negation of Hypothesis: No women are waiting by a tram.
    Negation of Premise: No high fashion ladies wait outside a tram beside a crowd of people in the city.
    Premise: People waiting to get on a train or just getting off.
    Hypothesis: There are people waiting on a train.
    Negation of Hypothesis: There are no people waiting on a train.
    Negation of Premise: No people are waiting to get on a train or just getting off.
    Premise: A couple play in the tide with their young son.
    Hypothesis: The family is outside.
    Negation of Hypothesis: The family is not outside.
    Negation of Premise: A couple does not play in the tide with their young son.
    Premise: A man and a woman cross the street in front of a pizza and gyro restaurant.
    Hypothesis: Near a couple of restaurants, two people walk across the street.
    Negation of Hypothesis: Near a couple of restaurants, no two people walk across the street.
    Negation of Premise: No man or woman cross the street in front of a pizza and gyro restaurant.
    Premise: A woman in a green jacket with a hood over her head is looking towards a valley.
    Hypothesis: The woman is wearing green.
    Negation of Hypothesis: The woman is not wearing green.
    Negation of Premise: No woman in a green jacket with a hood over her head is looking towards a valley.
    Premise: An elderly man is drinking orange juice at a cafe.
    Hypothesis: An old man is enjoying a beverage at a cafe.
    Negation of Hypothesis: No old man is enjoying a beverage at a cafe.
    Negation of Premise: No elderly man is drinking orange juice at a cafe.
    Premise: A couple holding hands is walking down a street.
    Hypothesis: People are holding hands and walking.
    Negation of Hypothesis: No people are holding hands and walking.
    Negation of Premise: No couple holding hands is walking down a street.
    Premise: A young woman is trying to stick her foot in a fountain.
    Hypothesis: A woman is near a fountain.
    Negation of Hypothesis: No woman is near a fountain.
    Negation of Premise: No young woman is trying to stick her foot in a fountain.
    Premise: Two people pose for the camera.
    Hypothesis: People pose for a picture.
    Negation of Hypothesis: No people pose for a picture.
    Negation of Premise: No two people pose for the camera.
    Premise: An older man wearing a salon drape getting a haircut.
    Hypothesis: A man gets a haircut.
    Negation of Hypothesis: No older man wearing a salon drape is getting a haircut.
    Negation of Premise: No man gets a haircut
    Premise: The children play happily in the park as they enjoy the sunny weather.
    Hypothesis: The park is full of joy.
    Negation of Hypothesis: The park is not full of joy.
    Negation of Premise: The children do not play happily in the park and do not enjoy the sunny weather.
    
    Take a closer look at these tricky cases; 'A' or 'An' needs to change to 'No' even if the sentence does not start with the subject, and also when two clauses exist in a single sentence.
    Premise: Under a blue sky with white clouds, a child reaches up to touch the propeller of a plane standing parked on a field of grass.
    Hypothesis: A child is reaching to touch the propeller of a plane.
    Negation of Hypothesis: Under a blue sky with white clouds, no child reaches up to touch the propeller of a plane standing parked on a field of grass.
    Negation of Premise: No child is reaching to touch the propeller of a plane
    Premise: A man in a green shirt is singing karaoke while a young woman with long brownish hair stands by and listens.
    Hypothesis: A man is singing a song.
    Negation of Hypothesis: No man is singing a song.
    Negation of Premise: No man in a green shirt is singing karaoke while no young woman with long browish hair stands by and listens.
    
    When a sentence starts with subjects specified with 'the' or 'this', verb should be negated rather than the subject.
    Premise: A woman with a green headscarf blue shirt and a very big grin.
    Hypothesis: The woman is very happy.
    Negation of Hypothesis: The woman is not very happy.
    Negation of Premise: No woman with a green headscarf blue shirt and a very big grin.
    Premise: The man walks among the large trees.
    Hypothesis: The man walks among trees.
    Negation of Hypothesis: The man does not walk among trees.
    Negation of Premise: The man does not walk among the large trees.
    
    Please note that given a compound sentence connecting two or more clauses with an 'and' or a comma, each of the clauses need to be negated, connected by an 'or'.
    Premise: Four people are near a body of water, two people walk on a sidewalk.
    Hypothesis: There are people outdoors.
    Negation of Hypothesis: There are no people outdoors
    Negation of Premise: No four people are near a body of water or no two people walk on a sidewalk.
    
    Construct Negation of Hypothesis and Negation of Premise by following the examples above.
    Given the condtion that Premise entails Hypothesis, Negation of Premise needs to entail Negation of Hypothesis.
    """
    return ask(instruction_negation, text)


def extract_sentences(text):
    lines = text.strip().split('\n')
    sentences = []
    for line in lines:
        if ':' in line:
            sentence = line.split(':', 1)[1].strip()
            sentences.append(sentence)

    return sentences



snli = pd.read_json('datasets/snli_1.0/snli_1.0_test.jsonl', lines=True)

## we need negation for only the contradiction & neutral label data
snli_contradiction = snli.loc[snli['gold_label']=='contradiction'].reset_index(drop=True).copy()
snli_neutral = snli.loc[snli['gold_label']=='neutral'].reset_index(drop=True).copy()

# Initialize an empty list to store results
results = []
for i, row in snli_contradiction.iterrows():  
        
    sentence = f"Premise: {row['sentence1']}\n Hypothesis: {row['sentence2']}"
    negation = get_negation(sentence)
    negation = extract_sentences(negation)
    
    not_p = negation[1]
    not_h = negation[0]
    
    pair_data = {
        'pairID': row['pairID'],
        'premise': row['sentence1'],
        'hypothesis': row['sentence2'],
        'premise_negated': not_p,
        'hypothesis_negated': not_h,
        'label': row['gold_label'],
    }
    results.append(pair_data)

with open('tasks/nli/negation/snli_test_contradiction_negated.json', 'w') as f:
    json.dump({'dataset_name': 'SNLI',
                'instances': results}, f, indent=4)
    
# Initialize an empty list to store results
results = []
for i, row in tqdm(snli_neutral.iterrows()):  
        
    sentence = f"Premise: {row['sentence1']}\n Hypothesis: {row['sentence2']}"
    negation = get_negation(sentence)
    negation = extract_sentences(negation)
    
    not_p = negation[1]
    not_h = negation[0]
    
    pair_data = {
        'pairID': row['pairID'],
        'premise': row['sentence1'],
        'hypothesis': row['sentence2'],
        'premise_negated': not_p,
        'hypothesis_negated': not_h,
        'label': row['gold_label'],
    }
    results.append(pair_data)

with open('tasks/nli/negation/snli_test_neutral_negated.json', 'w') as f:
    json.dump({'dataset_name': 'SNLI',
                'instances': results}, f, indent=4)

