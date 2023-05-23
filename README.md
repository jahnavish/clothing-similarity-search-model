# Clothing Similarity Search Model
This is an Clothing Similarity Search Model that receives text describing a clothing item and returns a ranked list of links to similar items from e-commerce websites.

The data is initially **web-scraped** from an e-commerce website, **pre-processed**, then the **similarity is computed** before the model **returns a ranked list of similar clothing items**. 

For pre-processing, the text data is checked for any null or duplicate values and then cleaned by first removing any spaces and hyphens in the descriptions of the clothing items, lowercasing the text and then applying stemming for text normalization.

Text vectorization is performed by two techniques, Term Frequency-Inverse Document Frequency (TF-IDF) and Bag of Words using the tools provided by the Python library 'scikit-learn'.

The similarity between the vectors is then computed using cosine similarity. A function is created which accepts the text description of the clothing item and based on the similarity computed, returns a json response of the ranked list of similar items from the dataset.

## To deploy this model and run it locally:

1. Clone the model (Run it in terminal or command prompt)

`git clone https://github.com/jahnavish/clothing-similarity-search-model.git`

2. Clone the model directly in Jupyter:

Open a Jupyter Notebook on your local machine

Run shell commands by adding ! as the prefix before your command

`!git clone https://github.com/jahnavish/clothing-similarity-search-model.git`

This will clone the repository directly in your Jupyter.
