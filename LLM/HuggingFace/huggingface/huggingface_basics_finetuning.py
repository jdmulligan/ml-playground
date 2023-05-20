'''
Basics of using the HuggingFace transformers/datasets/evaluate libraries
along w/pytorch to train and do inference for NLP tasks.

Based on:
 - https://huggingface.co/learn/nlp-course/chapter3/
'''

import tqdm

import torch
import transformers
import datasets
import evaluate

##########################################
def main():

    #----------------------------------------
    # Text classification using ...
    #----------------------------------------
    # We can train a model using a pre-existing dataset from the datasets library
    # Example: MRPC from the GLUE benchmark (https://huggingface.co/datasets/glue)
    #         5801 pairs of sentences, with binary label indicating whether the sentences are paraphrases
    # The datasets are downloaded to: ~/.cache/huggingface/datasets
    train_architecture_with_datasets()

#----------------------------------------
def train_architecture_with_datasets():

    # Load the model, dataset and create tokenizer
    checkpoint = "bert-base-uncased"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    dataset = datasets.load_dataset('glue', 'mrpc')
    print()
    print(f'Train the {checkpoint} model using the MRPC (GLUE) dataset...')
    print()
    print('The data has a pre-existing train/validation/test split:')
    print(f'  {dataset}')
    print()
    print('Example:')
    print(f'  {dataset["train"][0]}')
    print(f'  {dataset["train"].features}')
    print()

    # We can tokenize these pairs of sentences by using the Dataset.map() function
    #   which will apply a tokenizer function to each element of the dataset.
    # The tokenizer will add the token info to the original dataset dictionary
    # It does this efficiently:
    #   - tokenizer library parallelizes the tokenization
    #   - datasets library uses apache arrow to load only needed files to memory
    # We also create a data_collator object to use dynamic padding to pad each batch, rather than the entire dataset
    tokenize_function = lambda x: tokenizer(x['sentence1'], x['sentence2'], truncation=True)
    tokenized_dataset_dict = dataset.map(tokenize_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    print()
    print('Tokenized dataset:')
    print(f'  {tokenized_dataset_dict["train"][0]}')

    # Prepare tokenized data set for training with pytorch
    # We will strip away the original sentences, leaving only input_ids/token_type_ids/attention_mask/labels
    tokenized_dataset_dict = tokenized_dataset_dict.remove_columns(['idx', 'sentence1', 'sentence2'])
    tokenized_dataset_dict = tokenized_dataset_dict.rename_column('label', 'labels')
    tokenized_dataset_dict = tokenized_dataset_dict.with_format('torch')
    batch_size = 8
    dataloader_train = torch.utils.data.DataLoader(tokenized_dataset_dict['train'], 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   collate_fn=data_collator)
    print()
    print('We perform dynamical padding to each batch, for example:')
    for step, batch in enumerate(dataloader_train):
        print(f'  Batch {step} (batch_size, batch_length_padded):')
        for k, v in batch.items():
            print(f'    {k}: {v.shape}')
        if step >= 2:
            break
    print()

    # We can now train the model with pytorch
    print('Training model...')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)    

    optimizer = transformers.AdamW(model.parameters(), lr=3e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(dataloader_train)
    lr_scheduler = transformers.get_scheduler('linear',
                                              optimizer=optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=num_training_steps,
    )
    print(f'num_training_steps: {num_training_steps}')

    progress_bar = tqdm.tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for i,batch in enumerate(dataloader_train):
            if i > 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update()

    # Evaluate on the validation set
    dataloader_val = torch.utils.data.DataLoader(tokenized_dataset_dict['validation'], 
                                      batch_size=batch_size, 
                                      collate_fn=data_collator
                                     )
    metric = evaluate.load('glue', 'mrpc')
    model.eval()
    for batch in dataloader_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        # Compute forward pass (disable storing of computation graph, since we won't need gradients)
        with torch.no_grad():
            outputs = model(**batch)

        # Associate output logits with predicted labels
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])

    validation_metric = metric.compute()
    print()
    print(f'validation_metrics: {validation_metric}')

#----------------------------------------
if __name__ == '__main__':
    main()