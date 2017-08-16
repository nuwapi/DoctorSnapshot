#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:02:46 2017

@author: nwang
"""

import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import tokenize
import gensim
from gensim import corpora, models, similarities
import logging
import pickle

reviews = pd.read_csv("/home/nwang/proj/data/intermediate/SF_reviews.csv")
doctors = pd.read_csv("/home/nwang/proj/data/final/doctor_speadsheet_step2.csv")
review_main_text_original = reviews["review main text"]
master_string = '/biz/hayward-l-eubanks-hawthorne/biz/mervin-low-md-newport-beach/biz/davis-j-randall-md-marina-del-rey/biz/fisher-danelle-md-los-angeles/biz/woods-michelle-a-dds-los-angeles/biz/ullman-henry-md-beverly-hills/biz/keyvan-yousefi-md-los-angeles/biz/cha-john-y-dpm-inglewood/biz/david-aguilar-g-md-downey/biz/owens-g-stephen-md-glendale/biz/etie-moghissi-md-marina-del-rey/biz/mario-e-paz-dds-beverly-hills/biz/glenn-lipton-md-marina-del-rey/biz/andrew-yun-md-santa-monica/biz/pack-john-joseph-md-marina-del-rey/biz/kawilarang-harry-dds-huntington-park/biz/pegah-pourrahimi-dds-thousand-oaks/biz/leitner-physical-therapy-and-pilates-mar-vista/biz/foster-s-oliver-dpm-los-angeles/biz/mark-youssef-md-santa-monica/biz/william-chan-dds-santa-monica/biz/stephanie-duncan-garcia-do-miami/biz/kishibay-john-s-dmd-santa-monica/biz/bala-s-chandrasekhar-md-arcadia-2/biz/kevin-justus-md-facog-los-angeles/biz/eduardo-besser-md-culver-city/biz/howard-k-nam-md-facs-los-angeles/biz/satinder-j-s-bhatia-md-facc-fccp-beverly-hills/biz/rose-robert-m-md-beverly-hills/biz/deleaver-russell-margo-p-md-culver-city/biz/chiu-melvin-md-santa-monica/biz/hernandez-leopoldo-g-md-los-angeles/biz/belcher-gregory-md-saratoga/biz/sellman-john-r-md-santa-monica/biz/park-howard-h-dmd-md-santa-monica/biz/michael-estes-md-santa-monica/biz/orecklin-james-md-santa-monica/biz/phillips-albert-md-santa-monica-2/biz/jason-s-cohen-md-los-angeles-2/biz/feltman-joan-md-santa-monica/biz/sakhai-yussef-md-culver-city/biz/nora-m-hansen-md-chicago/biz/weiss-jeanne-m-md-santa-monica/biz/waring-graham-b-md-santa-monica/biz/seaside-medical-practice-nasimeh-yazdani-md-santa-monica/biz/z-joseph-wanski-md-los-angeles-2/biz/sanaz-khoubnazar-dmd-santa-monica/biz/assil-eye-institute-beverly-hills/biz/sajedi-ebrahim-md-santa-monica/biz/ralph-d-buoncristiani-dds-santa-monica/biz/daneshgar-kevin-md-los-angeles/biz/kimberly-ebner-dds-pasadena/biz/david-ng-md-los-angeles/biz/paul-fox-md-los-angeles/biz/galitzer-michael-md-los-angeles/biz/steven-k-cook-dds-santa-monica/biz/inouye-katherine-k-md-los-angeles/biz/united-care-family-medical-center-los-angeles/biz/peyman-banooni-md-beverly-hills/biz/nadiv-samimi-md-west-los-angeles/biz/duncan-jan-w-md-los-angeles/biz/jennifer-chin-md-los-angeles/biz/mike-a-uyeki-md-los-angeles/biz/o-brien-walter-r-md-inc-los-angeles/biz/chester-f-griffiths-md-los-angeles/biz/jessica-wu-md-los-angeles/biz/shadan-safvati-md-los-angeles/biz/elizabeth-chase-md-pacific-plsds/biz/yvonne-bohn-md-los-angeles/biz/joseph-h-park-md-los-angeles/biz/dr-song-s-cho-md-los-angeles/biz/david-a-boska-md-los-angeles/biz/wolfe-r-peter-md-los-angeles/biz/nancy-blumstein-md-los-angeles/biz/ton-hoang-d-md-los-angeles/biz/mark-weissman-dpm-los-angeles-2/biz/burstein-marina-md-los-angeles/biz/violet-boodaghian-md-los-angeles/biz/gayane-ambartsumyan-md-ph-d-redondo-beach/biz/friedland-robert-md-beverly-hills/biz/kimberly-a-caldwell-dds-chicago/biz/tourage-soleimani-md-los-angeles/biz/michel-babajanian-md-facs-los-angeles/biz/robert-f-meth-md-los-angeles/biz/green-abe-md-a-professional-medical-corporation-los-angeles/biz/alan-rosenbach-md-los-angeles/biz/lisa-g-cook-md-los-angeles/biz/howard-c-mandel-md-los-angeles/biz/stuart-j-fischbein-md-los-angeles/biz/dye-robert-s-md-agoura-hills/biz/helen-vu-dds-los-angeles/biz/sternberg-dermatology-los-angeles/biz/janet-vafaie-md-faad-malibu/biz/maloney-vision-institute-los-angeles-2/biz/lindsay-kiriakos-md-los-angeles/biz/paul-first-md-los-angeles/biz/ramtin-massoudi-md-beverly-hills/biz/peiman-berdjis-md-los-angeles-2/biz/douglas-davies-md-los-angeles-2/biz/richard-m-salit-md-los-angeles/biz/shahab-mehdizadeh-md-beverly-hills/biz/jeffrey-h-sherman-md-los-angeles/biz/gold-richard-n-md-los-angeles/biz/jose-r-ganata-jr-md-los-angeles/biz/doan-yen-md-los-angeles/biz/sangdo-park-md-los-angeles/biz/lee-carlton-y-s-md-los-angeles/biz/brian-n-huh-md-los-angeles/biz/chang-bong-s-md-los-angeles/biz/chris-fitzgerald-md-beverly-hills/biz/derek-cheng-md-beverly-hills/biz/karen-r-sandler-do-beverly-hills/biz/haleh-roohipour-md-beverly-hills/biz/graham-m-woolf-md-los-angeles/biz/annette-j-gottlieb-md-beverly-hills/biz/robert-f-katz-md-beverly-hills/biz/cornerstone-of-southern-california-santa-ana-2/biz/arash-moradzadeh-md-beverly-hills/biz/henry-yampolsky-md-beverly-hills-2/biz/may-lin-tao-md-beverly-hills/biz/william-j-binder-md-beverly-hills/biz/aiache-adrien-md-beverly-hills/biz/panosian-claire-md-los-angeles/biz/tabsh-khalil-md-los-angeles/biz/cheryl-charles-md-beverly-hills/biz/david-kawashiri-md-beverly-hills/biz/alexander-sinclair-md-whittier/biz/lee-john-s-md-beverly-hills/biz/michael-j-groth-md-beverly-hills/biz/kimberly-j-lee-md-beverly-hills/biz/marc-mani-md-beverly-hills/biz/dr-jeannine-rahimian-md-los-angeles/biz/tamkin-douglas-b-md-los-angeles/biz/garber-elayne-k-md-los-angeles/biz/stacey-p-rosenbaum-md-beverly-hills/biz/felipe-l-chu-md-los-angeles/biz/justin-saliman-md-los-angeles/biz/bruce-b-mclucas-md-los-angeles/biz/parviz-fahimian-md-beverly-hills/biz/rachel-abuav-md-beverly-hills/biz/norman-lepor-md-beverly-hills/biz/perry-liu-md-beverly-hills-5/biz/dohad-suhail-md-beverly-hills/biz/charles-swerdlow-md-beverly-hills/biz/kadner-marshall-l-md-beverly-hills/biz/robin-t-w-yuan-md-beverly-hills/biz/hal-danzer-md-beverly-hills/biz/david-charles-rish-md-beverly-hills/biz/kamran-kalpari-md-west-hollywood/biz/robert-o-ruder-md-beverly-hills/biz/kathleen-valenton-md-beverly-hills/biz/payam-abrishami-md-agoura-hills/biz/giuliano-armando-md-west-hollywood/biz/edgardo-falcon-dds-los-angeles/biz/hansen-david-c-md-west-hollywood/biz/leyli-dehghan-dds-beverly-hills/biz/choi-myunghae-md-los-angeles-2/biz/michael-levy-md-phd-san-diego/biz/brenda-c-smith-md-south-pasadena/biz/gustavo-a-alza-m-d-los-angeles/biz/carol-walker-md-south-pasadena/biz/lawton-w-tang-md-pasadena/biz/richard-menendez-md-glendale/biz/caroline-min-md-pasadena/biz/girard-philip-m-md-pasadena/biz/shunpei-keith-iwata-md-pasadena/biz/kamyar-amini-md-northridge-2/biz/daniel-j-casper-md-facs-pasadena-5/biz/bryan-s-jick-md-facog-pasadena/biz/armine-sarkisian-md-glendale/biz/gene-rubinstein-md-faad-studio-city/biz/sandeep-kapoor-md-studio-city/biz/lily-lee-md-pasadena/biz/norman-chien-md-pasadena/biz/peter-m-rosenberg-md-pasadena/biz/waleed-w-shindy-md-pasadena/biz/khin-khin-oo-md-pasadena/biz/iskander-ray-n-md-glendale/biz/kamyar-ebrahimi-md-glendale/biz/esther-y-yoon-md-glendale/biz/rudy-quintero-md-rancho-cucamonga/biz/tyler-hendry-dds-glendale/biz/chang-stephen-s-md-glendale/biz/reza-babapour-md-beverly-hills/biz/steven-a-rabin-md-burbank/biz/jeffrey-korchek-md-los-angeles/biz/william-weissman-dds-san-fernando-valley/biz/dardashti-dan-md-north-hollywood/biz/philip-biderman-m-d-f-a-c-s-sherman-oaks/biz/simon-robert-md-sherman-oaks/biz/gilman-podiatry-a-prof-corp-encino/biz/david-s-fine-dds-encino/biz/mark-kramar-md-encino/biz/gaytan-michael-h-dds-encino/biz/shaden-sarafzadeh-md-eye-institute-of-southern-california-encino/biz/robert-kahn-rose-md-phd-encino/biz/warren-line-md-burbank/biz/tri-d-dao-md-burbank/biz/maurice-berkowitz-m-d-burbank/biz/miao-peter-v-w-md-sherman-oaks/biz/amir-bahadori-md-los-angeles/biz/harry-l-dougherty-jr-dds-ms-sherman-oaks/biz/michael-malamed-md-tarzana/biz/mehras-akhavan-md-sherman-oaks/biz/stephen-d-bresnick-md-encino/biz/george-h-sanders-md-encino/biz/irina-ganelis-md-encino-2/biz/daws-douglas-a-dds-glendale/biz/bert-m-kaufman-dds-woodland-hills/biz/russo-joie-danielle-md-tarzana/biz/auerbach-joel-md-encino/biz/pean-marie-therese-md-tarzana-2/biz/morris-mesler-md-los-angeles/biz/fawaz-faisal-md-burbank/biz/lawrence-d-tran-md-burbank-2/biz/active-chiropractic-center-sherman-oaks/biz/alison-mann-encino/biz/raymond-bailey-md-tarzana/biz/raffi-margossian-dds-msd-burbank/biz/silberstein-taaly-md-tarzana/biz/robert-h-barnhard-md-tarzana-2/biz/irving-klasky-md-tarzana/biz/carlos-a-flores-md-burbank/biz/gevorkyan-hakop-md-burbank/biz/allan-l-kurtz-md-woodland-hills/biz/ebrahim-hakimian-md-van-nuys/biz/snyder-stephen-j-md-van-nuys/biz/friedman-marc-j-md-van-nuys/biz/reiche-andrea-md-van-nuys/biz/garbis-apelian-md-los-angeles/biz/mohammad-zareh-dds-riverside/biz/gingold-robin-md-west-hills/biz/michael-j-roberts-md-west-hills/biz/blair-s-kranson-md-canyon-country/biz/guy-massry-md-beverly-hills/biz/joana-tamayo-md-glendale/biz/jan-yuo-md-inc-montrose/biz/bloomfield-ellie-md-glendale/biz/mantell-alan-m-md-glendale/biz/kathryn-s-iwata-md-glendale/biz/scheinhorn-jeannine-md-glendale/biz/robert-m-miller-md-west-hills/biz/plance-donald-montrose/biz/jasmine-yun-md-west-hills/biz/gregory-c-yu-md-la-canada-flintridge/biz/habib-shehnaz-n-md-panorama-city/biz/leon-zonia-md-mission-hills/biz/charles-d-goodman-md-northridge/biz/michael-l-potruch-md-los-angeles/biz/moshe-lewis-md-san-francisco/biz/cesar-vega-m-d-northridge/biz/kayekjian-garabed-md-northridge/biz/darush-alan-md-apc-northridge/biz/yamaguchi-michael-md-terra-linda-pediatrics-san-rafael/biz/feldman-stefan-dpm-podtrst-northridge/biz/gabriel-aslanian-dds-md-northridge/biz/kling-michael-od-san-diego/biz/marshall-larry-j-md-lakeside/biz/childrens-primary-care-medical-group-la-jolla-3/biz/marianne-rochester-md-san-diego/biz/sanjay-ghosh-md-faans-san-diego/biz/perry-t-mansfield-md-san-diego/biz/michael-j-o-leary-md-san-diego/biz/karen-e-anderson-dpm-la-mesa/biz/tq-chiropractic-san-diego/biz/stacy-e-hulley-m-d-san-diego/biz/rita-j-feghali-m-d-san-diego/biz/richard-a-kaplan-m-d-san-diego/biz/jason-r-brown-dds-san-diego/biz/lin-cynthia-md-san-diego/biz/thoene-michael-j-md-el-cajon/biz/delois-j-bean-m-d-el-cajon/biz/jockin-yvette-m-md-san-diego/biz/kleid-jack-j-md-san-diego/biz/daniel-y-lee-md-san-diego/biz/richard-j-snyder-md-san-diego/biz/mirkarimi-morteza-san-diego-2/biz/milder-david-g-dds-md-san-diego-2/biz/boone-gary-md-san-diego/biz/robert-maywood-md-san-diego/biz/yorobe-edwin-m-md-san-diego/biz/robert-b-jacob-dds-san-diego/biz/juma-saad-md-encinitas/biz/david-kupfer-md-facs-san-diego/biz/paul-neustein-md-poway/biz/bridge-stephen-s-md-san-diego/biz/jeffrey-h-dysart-md-san-diego/biz/vidush-p-athyal-m-d-san-diego/biz/marc-k-rubenzik-m-d-san-diego/biz/eric-macy-m-d-san-diego/biz/grant-b-neifeld-m-d-san-diego/biz/patrick-c-watson-d-o-san-diego-2/biz/laura-a-mcmillan-m-d-san-diego/biz/ulrika-b-jansson-schumacher-m-d-san-diego/biz/sanjay-dhir-dds-san-diego/biz/sajben-nancy-l-md-la-jolla/biz/marc-e-kramer-md-san-diego/biz/bakst-isaac-md-san-diego/biz/jody-corey-bloom-md-la-jolla/biz/wendy-m-buchi-md-san-diego/biz/valerie-v-gafori-md-san-diego/biz/benito-villanueva-md-san-diego/biz/ronald-j-edelson-md-san-diego/biz/james-a-davis-md-san-francisco-2/biz/lam-chittaphong-dds-san-diego/biz/kj-ben-kim-dds-san-diego/biz/george-madany-md-san-diego/biz/james-p-tasto-dds-san-diego/biz/seibert-chiropractic-poway/biz/rivera-marcelo-r-md-int-med-poway'

review_main_text_list = []
for i in range(0, len(reviews)):
    paragraphs = reviews.loc[i]["review main text"]
    sentences = tokenize.sent_tokenize(paragraphs)
    for sentence in sentences:
        review_main_text_list.append(sentence)
                
# Set up tokenizer.
tokenizer = RegexpTokenizer(r'\w+')
# Set up stop words.
stop = set(stopwords.words('english'))
# Set up stemmer.
p_stemmer = PorterStemmer()
# Set up logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''
!! STEP 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
all_words = []
all_sentences = []
cleaned_up_review_list = []
for sentence in review_main_text_list:
    # Tokenization
    raw = sentence.lower()
    tokens = tokenizer.tokenize(raw)
    #print(tokens)
    
    # Get stop words
    j = 0
    while j < len(tokens):
        if tokens[j] in stop:
            #print (tokens[j], "del") 
            del tokens[j]
        else:
            #print (tokens[j], "save")
            j += 1
    #print(tokens)

    cleaned_text = [p_stemmer.stem(i) for i in tokens]
    #print(cleaned_text)
    
    all_sentences.append(cleaned_text)
    for token in cleaned_text:
        all_words.append(token)

# Generate corpus.
dictionary = gensim.corpora.Dictionary(all_sentences)
corpus = [dictionary.doc2bow(word) for word in all_sentences]

# Train!
lsa_model = models.LsiModel(corpus, id2word=dictionary, num_topics=1000)

# Use t-SNE
all_words_unique = nltk.FreqDist(all_words).most_common(4100)

#most_popular_vec = []
#for index1 in range(0, num_category):
#    for index2, item2 in enumerate(most_popular[index1]):
#        most_popular_vec.append(list(model.wv[item2]))
#    
#import numpy as np
#from sklearn.manifold import TSNE
#tsne_model = TSNE(n_components=2, random_state=0, perplexity=8.0)
#X = np.array(most_popular_vec)
#np.set_printoptions(suppress=True)
#tsne_result = tsne_model.fit_transform(X) 

all_words_unique_vec = []
all_words_unique_word = []
for index2, item2 in enumerate(all_words_unique):
    all_words_unique_vec.append([ x[1] for x in lsa_model[dictionary.doc2bow(["see"])] ])
    all_words_unique_word.append(item2[0])
    
import numpy as np
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, random_state=10, perplexity=100.0)
X = np.array(all_words_unique_vec)
np.set_printoptions(suppress=True)
tsne_result = tsne_model.fit_transform(X) 


import matplotlib.pyplot as plt
num_category = 11
most_popular = [
["care","patient","time","great","alway","recommend","staff","year","offic","best","question","good","help","realli"],
["insur","pay","compani","cover","charg","cost","paid","pocket","price","offic","payment","medic","servic","amount","claim"],
["call", "appoint", "offic", "get", "back", "time", "phone", "week", "told", "hour", "schedul", "wait", "said", "ask"],
["dentist", "teeth", "procedur", "staff", "pain", "tooth", "go", "recommend", "need", "dental", "feel", "would", "experi", "root", "wisdom"],
["surgeri","knee","surgeon","mri","injuri","shoulder","orthoped","month","hand","perform","bone"],
["babi","son","daughter","husband","pregnanc","deliv","birth","kid","first","ob","hospit","old","year","deliveri"],
["allergi","test","shot","allergist","food","year","prick","sinu","scratch","sick","cat","eczema","spray","steroid"],
["skin","dermatologist","acn","face","prescrib","cream","want","tri","csi"],
["eye","surgeon","done","consult","result","lasik","post","vision","perform"],
["breast","reconstruct","hand","recoveri","heal","reduct","remov","bone","thumb","implant","grind","broheen"],
["botox","kidney","treatment","western","inject","urologist","urolog"]
]

x = []
y = []
n = []

#counter = 0
#for index1 in range(0,num_category):
#    x.append([])
#    y.append([])
#    n.append([])
#    for index2, item2 in enumerate(most_popular[index1]):
#        x[index1].append(tsne_result[counter][0])
#        y[index1].append(tsne_result[counter][1])
#        n[index1].append(item2)
#        counter += 1

for index1 in range(0, num_category):
    x.append([])
    y.append([])
    n.append([])
    for index2, item2 in enumerate(most_popular[index1]):
        try:
            index3 = all_words_unique_word.index(item2)
            x[index1].append(tsne_result[index3][0])
            y[index1].append(tsne_result[index3][1])
            n[index1].append(item2)
        except:
            pass
# Plot
fig, ax = plt.subplots(figsize=(6,6), dpi=150, facecolor="white")
ax.set_facecolor("white")
plt.style.use("fivethirtyeight")

#for j in range(0, 3):
#    for i, txt in enumerate(n[j]):
#        ax.annotate(txt, (x[j][i],y[j][i]))
        
ax.scatter(x[0], y[0], label="Positive comments", color='#E74C3C')
ax.scatter(x[1], y[1], label="Payment", color='#8E44AD')
ax.scatter(x[2], y[2], label="Appointments & visits", color='#3498DB')
#ax.scatter(x[3], y[3], label="Dental care", color='#1ABC9C')
#ax.scatter(x[4], y[4], label="Surgery", color='#27AE60')
#ax.scatter(x[5], y[5], label="Women's health", color='#F1C40F')
#ax.scatter(x[6], y[6], label="Allergy treatments", color='#F39C12')
#ax.scatter(x[7], y[7], label="Skin procedures", color='#DC7633')
#ax.scatter(x[8], y[8], label="Eye care", color='#A6ACAF')
#ax.scatter(x[9], y[9], label="Reconstructive surgery", color='#5D6D7E')
#ax.scatter(x[10], y[10], label="Urology treatments", color='#000000')

frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend()

