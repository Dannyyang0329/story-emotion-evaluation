# %% [markdown]
# # Data Preprocessing for HANNA dataset


# %% [markdown]
# ## Import Libraries

# %%
import re
import os
import nltk
import torch
import warnings
import pandas as pd

from tqdm import tqdm
from dataclasses import dataclass, field
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM


warnings.filterwarnings("ignore")

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

# %% [markdown]
# ## Story Preprocessing and Sentence Tokenization

# %% [markdown]
# Read the data from the file and preprocess it.
# * Remove unrelated columns
# * Remove rows with missing values
# * Rename human annotated columns
# * NLTK Tokenization

# %%
class DataProcessor:
    def __init__(self, raw_csv_filepath: str, processed_csv_filepath: str):
        # download nltk resources
        nltk.download('punkt_tab')

        self.raw_csv_filepath = raw_csv_filepath
        self.processed_csv_filepath = processed_csv_filepath
        self.raw_story_df = pd.read_csv(raw_csv_filepath)
        self.preprocessed_story_df = None
    
    def preprocessed_story(self):
        # remove unrelated columns
        preprocessed_story_df = self.raw_story_df.copy()
        unrelated_column_list = ['Human', 'Worker ID', 'Assignment ID', 'Work time in seconds', 'Name']
        preprocessed_story_df.drop(columns=unrelated_column_list, inplace=True)
        preprocessed_story_df.dropna(inplace=True)
        preprocessed_story_df.reset_index(drop=True, inplace=True)
        # rename human annotated columns
        exist_target_column_list = ['Relevance', 'Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']
        preprocessed_story_df.rename(columns={col: f'Human {col}' for col in exist_target_column_list}, inplace=True)
        # NLTK sentence tokenization
        preprocessed_story_df['Sentences'] = preprocessed_story_df['Story'].apply(lambda x: sent_tokenize(x))
        preprocessed_story_df['Sentences Length'] = preprocessed_story_df['Sentences'].apply(lambda x: len(x))

        self.preprocessed_story_df = preprocessed_story_df
    
    def save_preprocessed_story(self):
        self.preprocessed_story_df.to_csv(self.processed_csv_filepath, index=False)


# %% [markdown]
# Process the data and save it to a new file

# %%
processor = DataProcessor(
    raw_csv_filepath='../hanna_data/hanna_stories_annotations.csv',
    processed_csv_filepath='../hanna_data/preprocessed_hanna_data.csv'
)

processor.preprocessed_story()
processor.save_preprocessed_story()

# %%
processor.preprocessed_story_df.tail(3)

# %%
processor.preprocessed_story_df.info()

# %% [markdown]
# ## Emoion Intensity Regression using EmoLLMs

# %% [markdown]
# ### Set up model configuration

# %%
@dataclass
class ModelConfig:
    model_name: str = 'lzw1008/Emollama-chat-7b'
    batch_size: int = 12
    generation_config: dict = field(default_factory=lambda: {
        "temperature": 0.9,
        "top_k": 30,
        "top_p": 0.6,
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 256,
        "pad_token_id": 0
    })
    

model_config = ModelConfig() 

# %%
PLUTCHIK_EMOTION_LIST = [
    "anger",
    "anticipation",
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
]

# %% [markdown]
# ### Define the Emotion Intensity Regression Model

# %%
class EIRer():
    def __init__(self, model_config: ModelConfig, pluchik_emotion_list: list):
        self.model_config = model_config
        self.pluchik_emotion_list = pluchik_emotion_list

        # load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name,
            device_map='auto',
            cache_dir="cache",
            use_fast=False
        ) 
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.padding_side = 'left'
        model_config.pad_token_id = self.tokenizer.pad_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            device_map='auto',
            cache_dir="cache",
        )

    def extract_intensity_score(self, text) -> float:
        pattern = r'Intensity\s+Score:\s*([\d\.]+)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        else:
            return -1

    def get_prompt(self, sentence_list: list, emotion: str) -> list:
        prompt_template=f"""\
Task: Assign a numerical value between 0 (least {emotion}) and 1 (most {emotion}) to represent the intensity of emotion {emotion} expressed in the part of story.\n\
Text: <STORY>\n\
Emotion: {emotion}\n\
Intensity Score:\
        """
        prompt_list = []
        for i in range(len(sentence_list)):
            prompt = prompt_template
            prompt = prompt.replace("<STORY>", sentence_list[i])
            prompt_list.append(prompt)
        return prompt_list


    def inference(self, sentence_list: list, emotion_list: list) -> dict:
        self.model.eval()
        output_dict = {}
        for emotion in emotion_list:
            prompt_list = self.get_prompt(sentence_list, emotion)
            output_dict[emotion] = []
            for i in range(0, len(prompt_list), self.model_config.batch_size):
                batch = prompt_list[i : min(i+self.model_config.batch_size, len(prompt_list))]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True)
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                # model generate output
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **self.model_config.generation_config
                )
                responses = self.tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True,
                    space_between_special_tokens=False
                )
                # check if the output is valid
                for j, response in enumerate(responses):
                    intensity_score = self.extract_intensity_score(response)
                    output_dict[emotion].append(intensity_score)
                
        return output_dict
    

# %% [markdown]
# ### Infer the emotion intensity scores

# %%
scorer = EIRer(model_config, PLUTCHIK_EMOTION_LIST)

score_list = []
total_story_num = len(processor.preprocessed_story_df)
for i in tqdm(range(total_story_num)):
    sentence_list = processor.preprocessed_story_df.loc[i, 'Sentences']
    emotion_score_dict = scorer.inference(sentence_list, PLUTCHIK_EMOTION_LIST)
    score_list.append(emotion_score_dict)

scored_story_df = processor.preprocessed_story_df.assign(**pd.DataFrame(score_list))

# %% [markdown]
# ### Save the emotion intensity scores

# %%
scored_story_df.to_csv('../hanna_data/scored_hanna_data.csv', index=False)


