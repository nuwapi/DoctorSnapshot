# The DoctorSnapshot pipline

Main pipline
* Step 1: Retrieve San Francisco doctors' basic information from the [BetterDoctor API](https://developer.betterdoctor.com/).
* Step 2: Parse the doctors' information (JSON files) in step 1 and save them into a CSV file.
* Step 3: Some of the above doctors have an associated Yelp link, only keep those doctors and scrape Yelp for their reviews.
* Step 4: Parse the Yelp HTML files for doctors' reviews.
* Step 5: Use the review dataset built above to train [latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA) models.
* Step 6: Use the trained LDA models and existing sentiment analyzers to generate the doctors' snapshots.

Further exploration
* Step 7: Try topic modeling/word embeding model [latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA) and compare it to LDA.
* Step 8: Try word embedding model [word2vec](https://en.wikipedia.org/wiki/Word2vec) and compare it to LDA.

Overall, LDA is a better fit for topic modeling compared to LSA and word2vec for this dataset.
