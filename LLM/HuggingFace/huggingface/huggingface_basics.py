'''
Basics of the pipeline for the HuggingFace transformers library

Based on:
 - https://huggingface.co/learn/nlp-course/chapter2/
'''

import transformers
import torch

##########################################
def main():

    # Text classification
    prompts = ['I love my pets.',
               'My pets are okay.',
               'My pets are annoying.',
               'I have a cat and a dog.',
               'My pets are hungry.'
              ]

    # Let's specify a model checkpoint
    # The model checkpoint will be downloaded and cached locally.
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
    checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

    # One can use a pre-existing pipeline to do a specified task
    print()
    print(f'Using existing pipeline...')
    print()
    print(f'Model: {checkpoint}')
    classifier = transformers.pipeline('sentiment-analysis',
                                       model=checkpoint)
    result = classifier(prompts) 
    print('The model found the following sentiments:')
    [print(f'  {prompts[i]} -- {result[i]["label"]} (score={result[i]["score"]:.3f})') for i in range(len(prompts))]
    print()

    # Alternately, we can perform the three steps of the pipeline manually:
    print(f'Using manual pipeline...')
    print()
    print(f'Model: {checkpoint}')

    # (1) Preprocessing: tokenization
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

#----------------------------------------
if __name__ == '__main__':
    main()