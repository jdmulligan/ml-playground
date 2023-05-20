'''
Basics of using the HuggingFace transformers library to use existing NLP models.

Based on:
 - https://huggingface.co/learn/nlp-course/chapter2/
'''

import torch
import transformers

##########################################
def main():

    #----------------------------------------
    # Text classification using BERT
    #----------------------------------------
    prompts = ['I love my pets.',
               'My pets are okay.',
               'My pets are annoying.',
               'I have a cat and a dog.',
               'My pets are hungry.'
              ]

    # Let's specify a model checkpoint
    # The model checkpoint will be downloaded and cached locally at ~/.cache/huggingface
    #  - Architecture: config.json 
    #  - Weights: pytorch_model.bin
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

    # One can use a pre-existing pipeline to do a specified task
    preexisting_pipeline(prompts, checkpoint)

    # Alternately, we can perform the three steps of the pipeline manually:
    manual_pipeline(prompts, checkpoint)

#----------------------------------------
def preexisting_pipeline(prompts, checkpoint):
    print()
    print(f'Using existing pipeline...')
    print()
    print(f'Model: {checkpoint}')
    classifier = transformers.pipeline('sentiment-analysis',
                                       model=checkpoint)
    result = classifier(prompts) 
    print('The model found the following sentiments:')
    [print(f'  {prompts[i]} -- {result[i]["label"]} (score={result[i]["score"]:.3f})') for i in range(len(prompts))]

#----------------------------------------
def manual_pipeline(prompts, checkpoint):
    print()
    print(f'Using manual pipeline...')
    print()
    print(f'Model: {checkpoint}')

    # (1) Preprocessing: tokenization
    #     Pad the inputs to a specified length
    #       - Note: The tokenizer will use an attention mask to skip attention mechanism 
    #               in the padded tokens, since we do not want to learn this context
    #     Truncate the inputs if longer than the max token window
    #     Return pytorch ("pt") tensors
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')

    # (2) Model: transformer + head
    #     The transformer outputs a high-dimensional vector for each input
    #     The head projects this into a lower-dimensional space to perform the given task
    model_without_head = transformers.AutoModel.from_pretrained(checkpoint)
    outputs_without_head = model_without_head(**inputs)
    print(f'Number of prompts: {outputs_without_head.last_hidden_state.shape[0]}')
    print(f'Token length: {outputs_without_head.last_hidden_state.shape[1]}')
    print(f'Embedding dimension: {outputs_without_head.last_hidden_state.shape[2]}')

    model_with_head = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs_with_head = model_with_head(**inputs)

    # (3) Postprocessing – convert logits (raw score) to probabilities
    predictions = torch.nn.functional.softmax(outputs_with_head.logits, dim=-1)
    print(f'Predictions: {predictions}')
    print(model_with_head.config.id2label)
    print()

#----------------------------------------
if __name__ == '__main__':
    main()