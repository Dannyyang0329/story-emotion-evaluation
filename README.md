# Auto Story Evaluation using Emotional Trajectories

## Dataset
96 story prompts are provided in the `hanna` dataset (data/hanna_llm_stories.csv). Human and 6 different LLM models listed below are asked to generate stories based on these prompts.
* Llama-7b
* Mistral-7b
* Beluga-13b
* OrcaPlatypus-13b
* LlamaInstruct-30b
* Platypus2-70b

NLTK tokenized stories are preprocessed and saved in
* data/tokenized_human_stories.csv
* data/tokenized_model_stories.csv

Using EmoLLMs(lzw1008/Emollama-chat-7b) to generate emotion intensity score for each token in the stories (sentence-level). 

The emotion intensity score is saved in
* data/human_story_emotion_scored.csv
* data/model_story_emotion_scored.csv