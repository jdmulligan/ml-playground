'''
Example of using the HuggingFace w/pytorch to train a semantic search algorithm:
find GitHub issues relevant for a user query

Based on:
 - https://huggingface.co/learn/nlp-course/chapter5/
'''

import os
import pandas as pd
import torch

import datasets
import transformers

##########################################
def main():

    #----------------------------------------
    # Use a pretrained embedding model
    #----------------------------------------
    checkpoint = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    model = transformers.AutoModel.from_pretrained(checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #----------------------------------------
    # Load dataset and compute embeddings
    #----------------------------------------
    outputdir = '/Users/jamesmulligan/.cache/huggingface/my_datasets/issues_dataset_with_embeddings'

    # If the embeddings have not been computed, we compute them
    if not os.path.exists(outputdir):
        print()
        print('Loading dataset and computing embeddings...')
        issues_dataset = load_dataset(outputdir, model, tokenizer, device)
        print('Done!')
        print()

    # If the embeddings have already been computed, we can just load them
    else:
        print()
        print(f'Embeddings have already been computed: {outputdir}')
        print('Loading from disk...')
        issues_dataset = datasets.load_from_disk(outputdir)
        print('Done!')
        print()

    #----------------------------------------
    # Perform search for a given query, using FAISS
    #---------------------------------------- 
    print('Querying dataset using FAISS...')
    issues_dataset.add_faiss_index(column="embeddings")

    # Construct a query and its embedding vector
    question = "How can I load a dataset offline?"
    print()
    print("=" * 50)
    print(f'Query: {question}')
    print("=" * 50)
    print()
    question_embedding = get_embeddings([question], model, tokenizer, device).cpu().detach().numpy()

    # Search for nearest neighbors
    n_results = 3
    scores, samples = issues_dataset.get_nearest_examples("embeddings", question_embedding, k=n_results)

    # Convert to pandas for easy viewing
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    for _, row in samples_df.iterrows():
        print("=" * 25)
        print(f' score={row.scores:.3f} ')
        print("=" * 25)
        print(row)
        #print(f"URL: {row.html_url}")
        #print(f"TITLE: {row.title}")
        #print(f"COMMENT: {row.comments}")
        print()

##########################################
def load_dataset(outputdir, model, tokenizer, device):

    # Get existing dataset of github issues
    issues_dataset = datasets.load_dataset("lewtun/github-issues", split="train")
    # Remove entries that correspond to pull requests rather than issues
    issues_dataset = issues_dataset.filter(lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0))
    # Only keep the columns we need
    columns = issues_dataset.column_names
    columns_to_keep = ["title", "body", "html_url", "comments"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    issues_dataset = issues_dataset.remove_columns(columns_to_remove)
    #print(issues_dataset)

    # Convert to pandas, to do some manipulations
    issues_dataset.set_format("pandas")
    df = issues_dataset[:]
    #print(df.columns)
    #print(df['comments'])
    # Each entry of the comments column is a list of comments -- let's explode to create a 
    # separate row in the df for each comment
    df_exploded = df.explode("comments", ignore_index=True)
    #print()
    #print(df_exploded.columns)
    #print(df_exploded['comments'])

    # Now let's convert it back to a HuggingFace dataset
    issues_dataset = datasets.Dataset.from_pandas(df_exploded)
    #print(issues_dataset)

    # Clean up a bit more by filtering out short comments
    issues_dataset = issues_dataset.map(lambda x: {"comment_length": len(x["comments"].split())})
    issues_dataset = issues_dataset.filter(lambda x: x["comment_length"] > 15)
    #print(issues_dataset)

    # Concatanate the title/body/comments
    issues_dataset = issues_dataset.map(lambda x: {"text": x["title"]+ " \n " + x["body"] + " \n " + x["comments"]})
    print(issues_dataset)

    #----------------------------------------
    # Compute embedding vector for each entry (using CLS pooling to get a single embedding for each sentence)
    # and add them to our dataset as numpy arrays
    #---------------------------------------- 
    print()
    print('Computing embeddings for issues dataset...')
    issues_dataset = issues_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"], model, tokenizer, device).detach().cpu().numpy()[0]}
    )   
    print('Done!')
    print()

    # Let's save the dataset with embeddings
    issues_dataset.save_to_disk(outputdir)
    return issues_dataset

##########################################
def get_embeddings(text, model, tokenizer, device):

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    model_output = model(**encoded_input)
    return cls_pooling(model_output)

##########################################
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

#----------------------------------------
if __name__ == '__main__':
    main()