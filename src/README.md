# The DoctorSnapshot pipeline

Main pipeline
* Step 1: Retrieve San Francisco doctors' basic information from the [BetterDoctor API](https://developer.betterdoctor.com/).
* Step 2: Parse the doctors' information (JSON files) retrieved in step 1 and save them into a CSV file.
* Step 3: Some of the above doctors have an associated [Yelp](https://www.yelp.com/) link, only keep those doctors and scrape Yelp for their reviews.
* Step 4: Parse the saved Yelp HTML files for doctors' reviews.
* Step 5: Use the review dataset built above to train [latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA) models.
* Step 6: Use the trained LDA models and existing sentiment analyzers to generate the doctors' snapshots.

Further exploration
* Step 7: Try topic modeling/word embedding model [latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA) and compare it to LDA.
* Step 8: Try word embedding model [word2vec](https://en.wikipedia.org/wiki/Word2vec) and compare it to LDA.

Overall, LDA is a better fit for topic modeling compared to LSA and word2vec for this dataset. Also [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) was attempted in step 5 and it does not work well with this dataset either.
