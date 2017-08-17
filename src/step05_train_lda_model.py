#!/usr/bin/env python3
# Created by Nuo Wang.
# Last modified on 8/17/2017.

# Required libraries.
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models, similarities
import logging
import pickle

### Step 1: Setting up.

# Load my dataset.
reviews = pd.read_csv("PATH/data/yelp_reviews.csv")
doctors = pd.read_csv("PATH/data/yelp_doctors.csv")

# The list to save the text bodies of all reviews.
review_main_text_list = []
# Add all review texts to the above list.
for i in range(0, len(reviews)):
    paragraphs = reviews.loc[i]["review main text"]
    review_main_text_list.append(paragraphs)
                
# Set up tokenizer.
tokenizer = RegexpTokenizer(r'\w+')
# Set up stop words.
stop = set(stopwords.words('english'))
# Set up stemmer.
p_stemmer = PorterStemmer()
# Set up logging for LDA in gensim.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

### Step 2: Clean up texts.

# List for the review texts that are tokenized, stop word deleted and stemmed.
cleaned_up_review_list = []
# For every review.
for document in review_main_text_list:
    # Use the lowercase of all letters.
    raw = document.lower()
    # Tokenization
    tokens = tokenizer.tokenize(raw)
    # Delete stop words.
    j = 0
    while j < len(tokens):
        if tokens[j] in stop:
            del tokens[j]
        else:
            j += 1
    # Stem each word.
    cleaned_text = [p_stemmer.stem(i) for i in tokens]
    # Add cleaned review text to list.
    cleaned_up_review_list.append(cleaned_text)

# Find words frequency.
all_words = []
for i in cleaned_up_review_list:
    for j in i:
        all_words.append(j)
fdist = nltk.FreqDist(all_words)
# Print the most common words.
print(fdist.most_common())

### Step 3: Delete more unwanted words and generate corpus.

# The words to ignore.
words_to_ignore = ["dr", "doctor"]
# Alternative versions of popular words to ignore.
# It turns out that if one ignores too many low frequency words, the LDA results are worse.
# words_to_ignore = ["dr", "doctor", "yelp", "dc", "dos", "dmd", "do", "dpm", "mbbch", "md", "od", "rpt", "california", "stanford", "ucla", "ucsf", "usc", "ucsd", "san", "francisco", "diego", "los", "angeles", "oakland", "beverly", "hills", "daly", "santa", "monica", "alamo", "solana", "beach", "poway", "del", "mar", "la", "jolla", "santee", "northridge", "rafael", "panorama", "canada", "flintridge", "glendale", "canyon", "westlake", "village", "riverside", "van", "nuys", "burbank", "tarzana", "encino", "oaks", "fernando", "pasadena", "rancho", "cucamonga", "hollywood", "institute", "york", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "korean", "filipino", "chinese", "embarcadero", "sacramento", "deborah", "irish", "http", "bob", "sculli", "claud", "washington", "meister", "eydelman"]
# words_to_ignore = ["acn","agoura","botox","canal","cardiologist","cedar","colonoscopi","dental","dentist","dermatologist","ear","eczema","eye","face","foot","fungal","hair","haircut","hip","http","jewish","knee","laser","lasik","mr","optometrist","orthoped","pediatrician","perm","physic","physician","salon","straight","surgeon","surgeri","www"]
# Also ignore all doctor names. This list is semi-automatically generated (code not shown here).
doctor_names = ['Ganata', 'Brian', 'Stein', 'Jack', 'Mitchell', 'Masami', 'Belcher', 'Rosanelli', 'Weissman', 'Nobel', 'Johnson', 'Michele', 'Shirazi', 'Francis', 'John', 'Vidush', 'Scott', 'Grogan', 'Sands', 'Navneet', 'Abrishami', 'Yun', 'Trott', 'Line', 'Sumati', 'Binstock', 'Ulrika', 'Yousefi', 'Ma', 'Guy', 'Mullens', 'Green', 'Kelly', 'Schiller', 'Ullman', 'Eric', 'Podolin', 'Vartany', 'Sanaz', 'Kashani', 'Collins', 'Jose', 'Mehdizadeh', 'Tiller', 'Baron', 'Maryam', 'Hill', 'Flynn', 'Cecilio', 'Kishibay', 'Stephane', 'Felicia', 'Enayati', 'M', 'Caroline', 'Fahimian', 'Glenn', 'Richard', 'Lam', 'Bakshian', 'Besser', 'Meth', 'Nassos', 'Walker', 'Pena-Robles', 'Forouzesh', 'Rochester', 'Friedman', 'Tri', 'Robert', 'Parnaz', 'Seibert', 'Sterman', 'Hakimian', 'Anupam', 'Rivera', 'Ardizzone', 'Chun', 'Vu', 'Eduardo', 'Lindsay', 'Dave', 'Mirabadi', 'Benbow', 'Fidel', 'Akizuki', 'Carlos C.', 'Wong', 'Rudy', 'Agnes', 'Venuturupalli', 'Fox', 'Urusova', 'Kourosh', 'Potruch', 'Jane', 'Jerry', 'Veronique', 'Hariri', 'Griffiths', 'Wadhwa', 'Maria', 'Stone', 'Lofthus', 'Metaxas', 'Cho', 'Fitzgerald', 'Larry', 'Ellie', 'Armine', 'Virginia', 'Kapla', 'Cohen', 'Yussef', 'Buoncristiani', 'Miao', 'Neidlinger', 'Sajben', 'Gladstein', 'Hillary', 'Samuel', 'Loraine', 'Hakop', 'Benito', 'Mikus', 'Mark', 'Inna', 'Pham', 'Luftman', 'Jacobitz', 'Fine', 'Yuo', 'Annette', 'Miller', 'Shadan', 'Bokosky', 'Sarosy', 'Kathleen', 'Lawrence', 'Boodaghians', 'Tessler', 'Jiu', 'Sandy', 'Babak', 'Michel', 'Yampolsky', 'Roykh', 'Carl', 'Berty', 'Liu', 'Tao', 'Stacy', 'Shahab', 'Catherine', 'Samimi', 'Crane', 'Violet', 'Cynthia', 'Smith', 'Schumacher', 'Kaufman', 'Carlton', 'Friedland', 'Belaga', 'Pourrahimi', 'Montgomery', 'Oo', 'Etie', 'Lakshmi', 'Henderson', 'Shaun', 'Chou', 'Antigone', 'Woods', 'Redlin', 'Yang', 'Carey', 'Pack', 'Sirius', 'Huh', 'Tamara', 'Leon', 'Deleaver-Russell', 'Neifeld', 'Katz', 'Heidenfelder', 'Melanie', 'Mansour', 'Stephen', 'Stern', 'Buchi', 'Yvonne', 'Henry', 'Aurasteh', 'Cafaro', 'Dillingham', 'Obrien', 'Putnam', 'Hyver', 'Ghada', 'Burstein', 'Winchell', 'Peggy', 'Kyoko', 'Garcia', 'Morganroth', 'Shankar', 'Jonah', 'Yorobe', 'Donald', 'Chow', 'Cepkinian', 'Hamidi', 'Chittaphong', 'Snibbe', 'Gennady', 'Tim', 'Joie', 'Rosenbaum', 'Lamont', 'Margossian', 'Silverman', 'Hoosik', 'Lewis', 'Le', 'Justine', 'Linda', 'Maureen', 'Dadvand', 'Mirkarimi', 'Quintero', 'Madany', 'Apelian', 'Ken', 'Berman', 'Corey Bloom', 'Kutzscher', 'Gafori', 'Gevorkyan', 'Akagi', 'Markison', 'Yabumoto', 'Valenton', 'Amir', 'Degolia', 'Kruse', 'Davidson', 'Ahluwalia', 'George', 'Elisa', 'Stephanie', 'Babajanian', 'Schultz', 'Sonal', 'Mohammad', 'Thoene', 'Wan', 'Babapour', 'Richards', 'Minkowsky', 'Galitzer', 'Myunghae', 'Vadim', 'Hess', 'Pegah', 'Kawashiri', 'Jay', 'Kram', 'Aguilar', 'Paiement', 'Theodore', 'Oleary', 'Izumi', 'Kathryn', 'Robin', 'Edgardo', 'Yoon', 'Ashish', 'Mike', 'Farinoush', 'Kimberly', 'Hornbrook', 'Betsy', 'Isaac', 'Massoudi', 'Gottlieb', 'Falcon', 'Athyal', 'Chu', 'Mynsberge', 'Helen', 'Collin', 'Ambartsumyan', 'Stacey', 'Romeo', 'Goei', 'Colbert', 'Hong', 'Brar', 'Mario', 'Levin', 'Charles', 'Fischbein', 'Meghan', 'Navab', 'Luis', 'Xiang', 'Orecklin', 'Banooni', 'Girard', 'Abe', 'Bhoot', 'Ebrahimi', 'Saliman', 'Joseph', 'Koltzova-Rang', 'Fakhouri', 'Matt', 'Dhir', 'Ghosh', 'Walter', 'Mehras', 'Pudberry', 'Satinder', 'Mehranpour', 'Rajan', 'Mann', 'Chang', 'Juliana', 'Sid', 'Hernandez', 'Arretz', 'Gregory', 'Marion', 'Howell', 'Greg', 'Van', 'Akhavan', 'Kerry', 'Blechman', 'Cha', 'Leopoldo', 'Jeannie', 'Cook', 'Giuliano', 'Scheinhorn', 'Weber', 'Raymond', 'Kupfer', 'Leung', 'Sarvenaz', 'Tuft', 'Amanda', 'Chobanian', 'Sima', 'Marc', 'Susan', 'Minoo', 'Ali', 'Eubanks', 'Soliman', 'Diego', 'Galpin', 'Sandler', 'Cinque', 'Eastman', 'Low', 'Marino', 'Lily', 'Barzman', 'Anita', 'Melissa', 'Maloney', 'Davies', 'Khoubnazar', 'Heather', 'Rachel', 'Molayem', 'Nita', 'Dan', 'Alison', 'Anderson', 'Alexander', 'Juma', 'Pool', 'Gin', 'Ganjianpour', 'Foster', 'Marriott', 'Bakst', 'Colette', 'Rust', 'Rahimian', 'Malamed', 'Jones', 'Nancy', 'Perl', 'Chester', 'Jasmine', 'Jared', 'Ehmer', 'Kalpari', 'Nazareth', 'Cortland', 'Chase', 'Arash', 'Birns', 'Bill', 'Eugene', 'Everson', 'David', 'Berkowitz', 'Nguyentu', 'Morteza', 'Zareh', 'May', 'Christine', 'Abuav', 'Amanuel', 'Mahnaz', 'Gingold', 'Desiree', 'Bruce', 'Parvin', 'McLucas', 'Azer', 'Kassab', 'Mandel', 'Shamie', 'Panosian', 'Andrea', 'Jerome', 'Sanjay', 'Bhuva', 'Gayane', 'Baker', 'Asaf', 'Gaminchi', 'Tran', 'Mamta', 'Philippe', 'Tang', 'Peiman', 'Rosen', 'Shiell', 'Iskander', 'Stefan', 'Levy', 'Jessica', 'Fung', 'Bailey', 'Edmund', 'Borah', 'Hulley', 'Armando', 'Kambiz', 'Stuart', 'Youssef', 'Blumstein', 'Groth', 'Frederica', 'Karpman', 'Palm', 'Nguyen', 'Davis', 'Zapata', 'Aizuss', 'William', 'Amirtharajah', 'Darush', 'Judy', 'Dina', 'Hans', 'Andre', 'Gary', 'Regan', 'Christman', 'Riedler', 'Justus', 'Yulia', 'Allan', 'Miclau', 'Edwin', 'Scalise', 'Bardowell', 'Jeanne', 'Reid', 'Agata', 'Reiche', 'Laurie', 'Castillo', 'Abraham', 'Arnold', 'Levi', 'Pedro', 'Thomas', 'Paul', 'Boska', 'Doan', 'Ng', 'Moradzadeh', 'Wolfe', 'Khalil', 'Park', 'Liao', 'Timothy', 'Nikolaj', 'Sinclair', 'Karlsberg', 'Yelding-Sloan', 'Marcelo', 'Graham', 'Shih', 'Bhatia', 'Waleed', 'Foxman', 'Khosravi', 'Yee', 'Inouye', 'Joshua', 'Edelson', 'Najarian', 'Klasky', 'Knox', 'Vaughn', 'Boykoff', 'Teguh', 'Kwang', 'Oliver', 'Alex', 'Debra', 'Snunit', 'Irene', 'Auerbach', 'Kearney', 'Gupta', 'Royeen', 'Tristan', 'Binder', 'Min', 'Kevin', 'Monali', 'Golshani', 'Emrani', 'Rostker', 'Lieu', 'Lepor', 'Massry', 'Claire', 'Rita', 'Cotter', 'Drell', 'Zdzislaus', 'Schechter', 'Vail', 'Kind', 'Marilyn', 'Jockin', 'Mantell', 'Kramer', 'Shaden', 'Sherman', 'Rafael', 'Holly', 'Jick', 'Uyeki', 'Ginsberg', 'Gold', 'Afshine', 'Casper', 'Khin', 'Hofstadter', 'Safvati', 'Suhail', 'Nick', 'Krames', 'Afshin', 'Myers', 'Sawsan', 'Meng', 'Mayer', 'Silberstein', 'Barkley', 'Kamanine', 'Warren', 'Shindy', 'Aslanian', 'Dembo-Smeaton', 'Prathipati', 'Rosenberg', 'Jeremy', 'Cepeda', 'Jade', 'Barnhard', 'Duong', 'Habib', 'Eisenhart', 'Tamayo', 'Goodman', 'Danelle', 'Christopher', 'Darakjian', 'Hal', 'Marquez', 'Moshe', 'Harry', 'Philip', 'Shirley', 'Alen', 'Hsu', 'Tu', 'Renee', 'Kiriakos', 'Estes', 'Kayekjian', 'Lynn', 'Kim', 'Perry', 'Tachdjian', 'Marnell', 'Karen', 'Sosa', 'Vahan', 'Talreja', 'Estwick', 'Floyd', 'Nilesh', 'Adrien', 'Yamaguchi', 'Tamer', 'Oechsel', 'Sandeep', 'Tabsh', 'Bowden', 'Bohn', 'Day', 'Edward', 'James', 'Zonia', 'Mani', 'Young', 'Stamper', 'Simon', 'Dougherty', 'Touradge', 'Otoole', 'Bresnick', 'Quan', 'Leslie', 'Sellman', 'Matthew', 'Donna', 'Swerdlow', 'Daphne', 'Mohamed', 'Pigeon', 'Marshall', 'Sawusch', 'Newcomer', 'Ganelis', 'Dysart', 'Hamilton', 'Russo', 'Lavi', 'Hayward', 'Cheryl', 'Jessely', 'Rabinovich', 'Lee', 'Mohammed', 'Rashti', 'Sarkisian', 'Factor', 'Jeannine', 'Laurence', 'Skoulas', 'Amarpreet', 'Bajaj', 'Beller', 'Villanueva', 'Lenzkes', 'Hu', 'Victor', 'Blair', 'Tarick', 'Alla', 'Dardashti', 'Tsoi', 'Sumeer', 'Mancherian', 'Levinson', 'Borookhim', 'Shervin', 'Flores', 'Bessie', 'Irina', 'Levine', 'Schofferman', 'Draupadi', 'Lipton', 'Epstein', 'Yen', 'Agbuya', 'Ruder', 'Mansfield', 'Rawat', 'Lin', 'Alessi', 'Eshaghian', 'Takahashi', 'Keyvan', 'Rashtian', 'Derrick', 'Arya Nick', 'Bert', 'Quock', 'Shukri', 'Flaherty', 'Kvitash', 'Man', 'Diamond', 'Choi', 'Lisa', 'Miremadi', 'Wendy', 'Liau', 'Lofquist', 'Shunpei', 'Douglas', 'Saito', 'Roya', 'Jenkin', 'Ralph', 'Dicks', 'Moghissi', 'De Luna', 'Michael', 'Nadiv', 'Schwanke', 'Kapoor', 'Starrett', 'Yip', 'Akash', 'Reese', 'Vera', 'Bridge', 'Feghali', 'Rosenbach', 'Jennifer', 'Brenda', 'Garg', 'Haleh', 'Chin', 'Yoo', 'Harold', 'Goodwin', 'Feltman', 'Shehnaz', 'Wu', 'Hendry', 'Emmanuel', 'Elena', 'Lakshman', 'Danzer', 'Maurice', 'Farnaz', 'Rose', 'Leitner', 'Khodabakhsh', 'Nam', 'Simoni', 'Parviz', 'Biderman', 'Snyder', 'Jacob', 'Anmar', 'Justin', 'Soleimani', 'Waring', 'Mueller', 'Fishman', 'Custis', 'Ann', 'Gilman', 'Nunes', 'Flach', 'Gores', 'Larian', 'Dana', 'Yokoyama', 'Dalwani', 'Chunbong', 'Grady', 'Carlos', 'Diana', 'Nora', 'Roberts', 'Elayne', 'Yvette', 'Weiss', 'Larisse', 'McMillan', 'Cesar', 'Kang', 'Chan', 'Gabriel', 'Rabin', 'Milder', 'Chenette', 'Lawton', 'Garabed', 'Malhotra', 'Char', 'Makassebi', 'Patel', 'Mesler', 'Eisele', 'Kenneth', 'Tamkin', 'Salit', 'Abhay', 'Keith', 'Sternberg', 'Wolff', 'Cortez', 'Rhee', 'Plance', 'Vincent', 'Pivo', 'Boone', 'Jonathan', 'Rosanna', 'Sangdo', 'Suzanne', 'Yu', 'Vega', 'Strom', 'Lau', 'Ben-Ozer', 'Hoyman', 'Bryan', 'Garbis', 'Hattori', 'Kahn-Rose', 'Macy', 'Woolf', 'Tamarin', 'Genen', 'Gaytan', 'Ramtin', 'Valerie', 'Atkin', 'Solomon', 'Fossett', 'Mahshid', 'Alikpala', 'Neustein', 'Tasto', 'Arjang', 'Ebrahim', 'Lief', 'Lara', 'Raul', 'Leah', 'Raffi', 'Darragh', 'Howard', 'Pedrotty', 'Serena', 'Chandrasekhar', 'Irving', 'Berdjis', 'Brown', 'Diggs', 'Sverdlov', 'Reza', 'Alza', 'Felipe', 'Yamada', 'Frederick', 'Jerrold', 'Orpilla', 'Peter', 'Barry', 'Sakhai', 'Alan', 'Kadner', 'Patrick', 'Jeffrey', 'Armstrong', 'Kleid', 'Paula', 'Tahani', 'Garber', 'Watson', 'Melvin', 'Gustavo', 'Roth', 'Smaili', 'Hoang', 'Devron', 'Daws', 'Tuan', 'Trojnar', 'Bong', 'Katherine', 'Nesari', 'Kawilarang', 'Ronald', 'Kamran', 'Gordon', 'Menendez', 'Bortz', 'Massey', 'Rubenzik', 'Alfred', 'Marina', 'Fawaz', 'Shafipour', 'Bloomfield', 'Feldman', 'Chua', 'Pouya', 'Peyman', 'Norman', 'Stefani', 'Yazdani', 'Sameer', 'Dohad', 'Kurtz', 'Molato', 'Refoa', 'Marie', 'Engel', 'Pamela', 'Caroll', 'Daneshgar', 'Sun', 'Haley', 'Valentina', 'Leonard', 'Maeck', 'Michelle', 'Roohipour', 'Faisal', 'Payam', 'Kramar', 'Kerman', 'Sherwin', 'Khoury', 'Garrick', 'Leyli', 'Wanski', 'Cardon', 'Pean', 'Assil', 'Bahadori', 'Andrew', 'Rodney', 'Chiu', 'Taaly', 'Remy', 'Fisher', 'Sharon', 'Melody', 'Bala', 'Armen', 'Mobasser', 'Joy', 'Nader', 'Beeve', 'Hammond', 'Vanhale', 'Cheung', 'Cheng', 'Kranson', 'Sloan', 'Delois', 'Silani', 'Wieder', 'Vafaie', 'Chien', 'Brandeis', 'Su', 'Gamache', 'Ray', 'Bean', 'Mohana', 'Guido', 'Starnes', 'Chong', 'Martin', 'Daniel', 'Biana', 'Schulman', 'Marianne', 'Randolph', 'Morris', 'Xilin', 'Bailony', 'Reynaldo', 'Caldwell', 'Song', 'Herbert', 'Saad', 'Elgan', 'Esther', 'Nasimeh', 'Paz', 'Greenberg', 'Hopper', 'Derek', 'Grant', 'Vlad', 'Kaplan', 'Amini', 'Albert', 'Kling', 'Benjamin', 'Sam', 'Cabrera', 'Chiu-Collins', 'Wolfson', 'Margo', 'Cowan', 'Chen', 'Payman', 'Rish', 'Sanders', 'Cameron', 'Owens', 'Phillips', 'Dao', 'Allison', 'Maywood', 'Elliot', 'Jody', 'Thaik', 'Korchek', 'Eng', 'Ton', 'Thuc', 'Nathan', 'Glen', 'Bickman', 'Reyes', 'Sarafzadeh', 'Hansen', 'Yuan', 'Nikole', 'Mervin', 'Aiache', 'Iwata', 'Considine', 'Tyler', 'Rodriguez', 'Dawn', 'Steven', 'Carol', 'E.', 'Kamyar', 'Tin', 'Jason', 'Moy', 'Duncan', 'Merilynn', 'Dye', 'Chaves', 'Sajedi', 'Strelkoff', 'Lattanza', 'Janet', 'Joan', 'Elizabeth', 'Weller', 'Swamy', 'Rupsa', 'Laura', 'Shu', 'Joana', 'Jan', 'Joel', 'Rubinstein', 'Co']
# List to store the cleaned up doctor names.
doctor_names_cleaned_up = []
for name in doctor_names:
    doctor_names_cleaned_up.append(p_stemmer.stem(name.lower()))
# Append cleaned up doctor names to the words to ignore.
words_to_ignore += doctor_names_cleaned_up

# Filter out the words that we want to ignore.
cleaned_up_review_list2 = list(cleaned_up_review_list)
for tokens in cleaned_up_review_list2:    
    # Delete specificed words.
    j = 0
    while j < len(tokens):
        if tokens[j] in words_to_ignore:
            del tokens[j]
        else:
            j += 1
    
# Generate dictionary and corpus from the remaining words.
dictionary = gensim.corpora.Dictionary(cleaned_up_review_list2)
corpus = [dictionary.doc2bow(word) for word in cleaned_up_review_list2]

# TF-IDF.
# TF-IDF does not work well for this dataset, it leads to worse results.
# I have a good explanation why, you can ask me.
# tfidf = models.TfidfModel(corpus)
# corpus = tfidf[corpus]

### Step 4: Train LDA model.

# Set up LDA model parameters.
no_of_topics = 15
passes_in = 100

# Train LDA model and save the model results.
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_of_topics, id2word = dictionary, passes=passes_in, alpha='asymmetric')
pickle.dump(ldamodel, open("PATH/model/lda.pickle", "wb"))
pickle.dump(dictionary, open("PATH/model/dictionary.pickle", "wb"))
pickle.dump(corpus, open("PATH/model/corpus.pickle", "wb"))

# Check resulting topics.
topic_list = ldamodel.print_topics(num_topics=no_of_topics, num_words=15)
for index, i in enumerate(topic_list):
    str1 = str(i[1])
    for c in "0123456789+*\".":
        str1 = str1.replace(c, "")
    str1 = str1.replace("  ", " ")
    print(str1)

########################################################
#### Using a toy corpus to show why TF-IDF doesn't work.
texts = [["dentist", "teeth", "insurance", "bill", "pizza"],
         ["dentist", "teeth", "appointment", "calll", "magician"],
         ["insurance", "bill", "appointment", "call", "fifteen"]]
dictionary_text = gensim.corpora.Dictionary(texts)
corpus_text = [dictionary_text.doc2bow(word) for word in texts]
tfidf_text = models.TfidfModel(corpus_text)
corpus_tfidf = tfidf_text[corpus_text]
ldamodel_text_1 = gensim.models.ldamodel.LdaModel(corpus_text, num_topics=4, id2word = dictionary_text, passes=200)
ldamodel_text_2 = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=4, id2word = dictionary_text, passes=200)

topic_list_1 = ldamodel_text_1.print_topics()
topic_list_2 = ldamodel_text_2.print_topics()
for index, i in enumerate(topic_list_1):
    str1 = str(i[1])
    for c in "0123456789+*\".":
        str1 = str1.replace(c, "")
    str1 = str1.replace("  ", " ")
    print(str1)
print()
for index, i in enumerate(topic_list_2):
    str1 = str(i[1])
    for c in "0123456789+*\".":
        str1 = str1.replace(c, "")
    str1 = str1.replace("  ", " ")
    print(str1)
