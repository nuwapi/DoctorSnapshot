# DoctorSnapshot
DoctorSnapshot is a machine learning-powered webapp that helps people quickly spot the doctors best to them through itemized doctor ratings and review highlights.

Useful links:
* [DoctorSnapshot website](http://doctorsnapshot.herokuapp.com/)
* [DoctorSnapshot presentation slides](http://slides.com/nwangpierse/doctorsnapshot_slides)
* [Running your copy of DoctorSnapshot](https://github.com/nuowang/DoctorSnapshot/tree/master/webapp)

## 1. Dataset
* 187 San Francisco doctors\*.
* 5088 doctor reviews from Yelp.
* 700,000 words.
* 10,000 unique words.

\* Currently, the DoctorSnapshot database only contains doctors in San Francisco and who has Yelp reviews. But the DoctorSnapshot model and workflow can be readily applied to any doctors and their reviews if data is available.

## 2. Itemized doctor ratings
Instead of a single rating for a doctor, e.g. X out of 5 stars, DoctorSnapshot extracts the commonly mentioned topics within a doctor's reviews, e.g. payment exprience at a clinic, and detects how many people are speaking positively when they mentioned these topics.

The "itemized rating" for a topic that appeared in a doctor's reviews is a number between 0 to 100 that represents the percentage of people that spoke positively on that topic.

## 3. Review highlights
The three most positive and three most negative sentences across all of the reviews of a doctor are selected as highlights for the whole review body.

## 4. DoctorSnapshot's model
DoctorSnapshot uses [latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) implemented in Python pakage [gensim](https://radimrehurek.com/gensim/) to extract topics from the reviews. And the sentiment of each review sentence is evaluated by VADER sentiment analyzer implemented in Python package [NLTK](http://www.nltk.org/).
