# DoctorSnapshot
DoctorSnapshot is a machine learning-powered webapp that helps users spot the doctors best for them by providing itemized doctor ratings and review highlights extracted from the doctor's reviews. Users can leave all of the tedious review readings to DoctorSnapshot and enjoy highly concentrated reports (doctors' snapshots) for quick decision making.

* [DoctorSnapshot website](http://doctorsnapshot.herokuapp.com/)
* [DoctorSnapshot presentation slides](http://slides.com/nwangpierse/doctorsnapshot_slides)
* [The DoctorSnapshot pipeline](https://github.com/nuowang/DoctorSnapshot/tree/master/src)
* [Running your copy of DoctorSnapshot](https://github.com/nuowang/DoctorSnapshot/tree/master/webapp)

The DoctorSnapshot project was completed in only 3 weeks, from ideation to deployment. Many future improvements are possible and desired. I will be glad to hear your suggestions!

## 1. Dataset
* 187 San Francisco doctors.
* 5088 doctor reviews from Yelp.
* 700,000 words.
* 10,000 unique words.

Currently, the DoctorSnapshot dataset only contains doctors in San Francisco and who has Yelp reviews. But the DoctorSnapshot model and workflow can be readily applied to any doctor and their reviews if such data is available.

The DoctorSnapshot dataset contains Yelp reviews, which cannot be shared publicly according to Yelp's terms of service.

## 2. Itemized doctor ratings
Instead of giving a single rating to a doctor, e.g. X out of 5 stars, DoctorSnapshot extracts the commonly mentioned topics within a doctor's reviews, e.g. payment exprience at the clinic, and detects how many people are speaking positively when they mentioned these topics.

The "itemized rating" for a topic that appeared in a doctor's reviews is a number between 0 to 100 that represents the percentage of people that spoke positively on that topic.

## 3. Review highlights
The three most positive and three most negative sentences across all of the reviews of a doctor are selected as highlights for the whole review body of the doctor.

## 4. The machine learning model
DoctorSnapshot uses [latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) implemented in Python package [gensim](https://radimrehurek.com/gensim/) to extract topics from doctor reviews. The sentiment of each review sentence is evaluated by the VADER sentiment analyzer implemented in Python package [NLTK](http://www.nltk.org/).
