#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:13:13 2017

@author: nwang
"""

import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models, similarities
import logging
import pickle

reviews = pd.read_csv("/home/nwang/proj/data/intermediate/SF_reviews.csv")
doctors = pd.read_csv("/home/nwang/proj/data/final/doctor_speadsheet_step2.csv")
review_main_text_original = reviews["review main text"]
master_string = '/biz/hayward-l-eubanks-hawthorne/biz/mervin-low-md-newport-beach/biz/davis-j-randall-md-marina-del-rey/biz/fisher-danelle-md-los-angeles/biz/woods-michelle-a-dds-los-angeles/biz/ullman-henry-md-beverly-hills/biz/keyvan-yousefi-md-los-angeles/biz/cha-john-y-dpm-inglewood/biz/david-aguilar-g-md-downey/biz/owens-g-stephen-md-glendale/biz/etie-moghissi-md-marina-del-rey/biz/mario-e-paz-dds-beverly-hills/biz/glenn-lipton-md-marina-del-rey/biz/andrew-yun-md-santa-monica/biz/pack-john-joseph-md-marina-del-rey/biz/kawilarang-harry-dds-huntington-park/biz/pegah-pourrahimi-dds-thousand-oaks/biz/leitner-physical-therapy-and-pilates-mar-vista/biz/foster-s-oliver-dpm-los-angeles/biz/mark-youssef-md-santa-monica/biz/william-chan-dds-santa-monica/biz/stephanie-duncan-garcia-do-miami/biz/kishibay-john-s-dmd-santa-monica/biz/bala-s-chandrasekhar-md-arcadia-2/biz/kevin-justus-md-facog-los-angeles/biz/eduardo-besser-md-culver-city/biz/howard-k-nam-md-facs-los-angeles/biz/satinder-j-s-bhatia-md-facc-fccp-beverly-hills/biz/rose-robert-m-md-beverly-hills/biz/deleaver-russell-margo-p-md-culver-city/biz/chiu-melvin-md-santa-monica/biz/hernandez-leopoldo-g-md-los-angeles/biz/belcher-gregory-md-saratoga/biz/sellman-john-r-md-santa-monica/biz/park-howard-h-dmd-md-santa-monica/biz/michael-estes-md-santa-monica/biz/orecklin-james-md-santa-monica/biz/phillips-albert-md-santa-monica-2/biz/jason-s-cohen-md-los-angeles-2/biz/feltman-joan-md-santa-monica/biz/sakhai-yussef-md-culver-city/biz/nora-m-hansen-md-chicago/biz/weiss-jeanne-m-md-santa-monica/biz/waring-graham-b-md-santa-monica/biz/seaside-medical-practice-nasimeh-yazdani-md-santa-monica/biz/z-joseph-wanski-md-los-angeles-2/biz/sanaz-khoubnazar-dmd-santa-monica/biz/assil-eye-institute-beverly-hills/biz/sajedi-ebrahim-md-santa-monica/biz/ralph-d-buoncristiani-dds-santa-monica/biz/daneshgar-kevin-md-los-angeles/biz/kimberly-ebner-dds-pasadena/biz/david-ng-md-los-angeles/biz/paul-fox-md-los-angeles/biz/galitzer-michael-md-los-angeles/biz/steven-k-cook-dds-santa-monica/biz/inouye-katherine-k-md-los-angeles/biz/united-care-family-medical-center-los-angeles/biz/peyman-banooni-md-beverly-hills/biz/nadiv-samimi-md-west-los-angeles/biz/duncan-jan-w-md-los-angeles/biz/jennifer-chin-md-los-angeles/biz/mike-a-uyeki-md-los-angeles/biz/o-brien-walter-r-md-inc-los-angeles/biz/chester-f-griffiths-md-los-angeles/biz/jessica-wu-md-los-angeles/biz/shadan-safvati-md-los-angeles/biz/elizabeth-chase-md-pacific-plsds/biz/yvonne-bohn-md-los-angeles/biz/joseph-h-park-md-los-angeles/biz/dr-song-s-cho-md-los-angeles/biz/david-a-boska-md-los-angeles/biz/wolfe-r-peter-md-los-angeles/biz/nancy-blumstein-md-los-angeles/biz/ton-hoang-d-md-los-angeles/biz/mark-weissman-dpm-los-angeles-2/biz/burstein-marina-md-los-angeles/biz/violet-boodaghian-md-los-angeles/biz/gayane-ambartsumyan-md-ph-d-redondo-beach/biz/friedland-robert-md-beverly-hills/biz/kimberly-a-caldwell-dds-chicago/biz/tourage-soleimani-md-los-angeles/biz/michel-babajanian-md-facs-los-angeles/biz/robert-f-meth-md-los-angeles/biz/green-abe-md-a-professional-medical-corporation-los-angeles/biz/alan-rosenbach-md-los-angeles/biz/lisa-g-cook-md-los-angeles/biz/howard-c-mandel-md-los-angeles/biz/stuart-j-fischbein-md-los-angeles/biz/dye-robert-s-md-agoura-hills/biz/helen-vu-dds-los-angeles/biz/sternberg-dermatology-los-angeles/biz/janet-vafaie-md-faad-malibu/biz/maloney-vision-institute-los-angeles-2/biz/lindsay-kiriakos-md-los-angeles/biz/paul-first-md-los-angeles/biz/ramtin-massoudi-md-beverly-hills/biz/peiman-berdjis-md-los-angeles-2/biz/douglas-davies-md-los-angeles-2/biz/richard-m-salit-md-los-angeles/biz/shahab-mehdizadeh-md-beverly-hills/biz/jeffrey-h-sherman-md-los-angeles/biz/gold-richard-n-md-los-angeles/biz/jose-r-ganata-jr-md-los-angeles/biz/doan-yen-md-los-angeles/biz/sangdo-park-md-los-angeles/biz/lee-carlton-y-s-md-los-angeles/biz/brian-n-huh-md-los-angeles/biz/chang-bong-s-md-los-angeles/biz/chris-fitzgerald-md-beverly-hills/biz/derek-cheng-md-beverly-hills/biz/karen-r-sandler-do-beverly-hills/biz/haleh-roohipour-md-beverly-hills/biz/graham-m-woolf-md-los-angeles/biz/annette-j-gottlieb-md-beverly-hills/biz/robert-f-katz-md-beverly-hills/biz/cornerstone-of-southern-california-santa-ana-2/biz/arash-moradzadeh-md-beverly-hills/biz/henry-yampolsky-md-beverly-hills-2/biz/may-lin-tao-md-beverly-hills/biz/william-j-binder-md-beverly-hills/biz/aiache-adrien-md-beverly-hills/biz/panosian-claire-md-los-angeles/biz/tabsh-khalil-md-los-angeles/biz/cheryl-charles-md-beverly-hills/biz/david-kawashiri-md-beverly-hills/biz/alexander-sinclair-md-whittier/biz/lee-john-s-md-beverly-hills/biz/michael-j-groth-md-beverly-hills/biz/kimberly-j-lee-md-beverly-hills/biz/marc-mani-md-beverly-hills/biz/dr-jeannine-rahimian-md-los-angeles/biz/tamkin-douglas-b-md-los-angeles/biz/garber-elayne-k-md-los-angeles/biz/stacey-p-rosenbaum-md-beverly-hills/biz/felipe-l-chu-md-los-angeles/biz/justin-saliman-md-los-angeles/biz/bruce-b-mclucas-md-los-angeles/biz/parviz-fahimian-md-beverly-hills/biz/rachel-abuav-md-beverly-hills/biz/norman-lepor-md-beverly-hills/biz/perry-liu-md-beverly-hills-5/biz/dohad-suhail-md-beverly-hills/biz/charles-swerdlow-md-beverly-hills/biz/kadner-marshall-l-md-beverly-hills/biz/robin-t-w-yuan-md-beverly-hills/biz/hal-danzer-md-beverly-hills/biz/david-charles-rish-md-beverly-hills/biz/kamran-kalpari-md-west-hollywood/biz/robert-o-ruder-md-beverly-hills/biz/kathleen-valenton-md-beverly-hills/biz/payam-abrishami-md-agoura-hills/biz/giuliano-armando-md-west-hollywood/biz/edgardo-falcon-dds-los-angeles/biz/hansen-david-c-md-west-hollywood/biz/leyli-dehghan-dds-beverly-hills/biz/choi-myunghae-md-los-angeles-2/biz/michael-levy-md-phd-san-diego/biz/brenda-c-smith-md-south-pasadena/biz/gustavo-a-alza-m-d-los-angeles/biz/carol-walker-md-south-pasadena/biz/lawton-w-tang-md-pasadena/biz/richard-menendez-md-glendale/biz/caroline-min-md-pasadena/biz/girard-philip-m-md-pasadena/biz/shunpei-keith-iwata-md-pasadena/biz/kamyar-amini-md-northridge-2/biz/daniel-j-casper-md-facs-pasadena-5/biz/bryan-s-jick-md-facog-pasadena/biz/armine-sarkisian-md-glendale/biz/gene-rubinstein-md-faad-studio-city/biz/sandeep-kapoor-md-studio-city/biz/lily-lee-md-pasadena/biz/norman-chien-md-pasadena/biz/peter-m-rosenberg-md-pasadena/biz/waleed-w-shindy-md-pasadena/biz/khin-khin-oo-md-pasadena/biz/iskander-ray-n-md-glendale/biz/kamyar-ebrahimi-md-glendale/biz/esther-y-yoon-md-glendale/biz/rudy-quintero-md-rancho-cucamonga/biz/tyler-hendry-dds-glendale/biz/chang-stephen-s-md-glendale/biz/reza-babapour-md-beverly-hills/biz/steven-a-rabin-md-burbank/biz/jeffrey-korchek-md-los-angeles/biz/william-weissman-dds-san-fernando-valley/biz/dardashti-dan-md-north-hollywood/biz/philip-biderman-m-d-f-a-c-s-sherman-oaks/biz/simon-robert-md-sherman-oaks/biz/gilman-podiatry-a-prof-corp-encino/biz/david-s-fine-dds-encino/biz/mark-kramar-md-encino/biz/gaytan-michael-h-dds-encino/biz/shaden-sarafzadeh-md-eye-institute-of-southern-california-encino/biz/robert-kahn-rose-md-phd-encino/biz/warren-line-md-burbank/biz/tri-d-dao-md-burbank/biz/maurice-berkowitz-m-d-burbank/biz/miao-peter-v-w-md-sherman-oaks/biz/amir-bahadori-md-los-angeles/biz/harry-l-dougherty-jr-dds-ms-sherman-oaks/biz/michael-malamed-md-tarzana/biz/mehras-akhavan-md-sherman-oaks/biz/stephen-d-bresnick-md-encino/biz/george-h-sanders-md-encino/biz/irina-ganelis-md-encino-2/biz/daws-douglas-a-dds-glendale/biz/bert-m-kaufman-dds-woodland-hills/biz/russo-joie-danielle-md-tarzana/biz/auerbach-joel-md-encino/biz/pean-marie-therese-md-tarzana-2/biz/morris-mesler-md-los-angeles/biz/fawaz-faisal-md-burbank/biz/lawrence-d-tran-md-burbank-2/biz/active-chiropractic-center-sherman-oaks/biz/alison-mann-encino/biz/raymond-bailey-md-tarzana/biz/raffi-margossian-dds-msd-burbank/biz/silberstein-taaly-md-tarzana/biz/robert-h-barnhard-md-tarzana-2/biz/irving-klasky-md-tarzana/biz/carlos-a-flores-md-burbank/biz/gevorkyan-hakop-md-burbank/biz/allan-l-kurtz-md-woodland-hills/biz/ebrahim-hakimian-md-van-nuys/biz/snyder-stephen-j-md-van-nuys/biz/friedman-marc-j-md-van-nuys/biz/reiche-andrea-md-van-nuys/biz/garbis-apelian-md-los-angeles/biz/mohammad-zareh-dds-riverside/biz/gingold-robin-md-west-hills/biz/michael-j-roberts-md-west-hills/biz/blair-s-kranson-md-canyon-country/biz/guy-massry-md-beverly-hills/biz/joana-tamayo-md-glendale/biz/jan-yuo-md-inc-montrose/biz/bloomfield-ellie-md-glendale/biz/mantell-alan-m-md-glendale/biz/kathryn-s-iwata-md-glendale/biz/scheinhorn-jeannine-md-glendale/biz/robert-m-miller-md-west-hills/biz/plance-donald-montrose/biz/jasmine-yun-md-west-hills/biz/gregory-c-yu-md-la-canada-flintridge/biz/habib-shehnaz-n-md-panorama-city/biz/leon-zonia-md-mission-hills/biz/charles-d-goodman-md-northridge/biz/michael-l-potruch-md-los-angeles/biz/moshe-lewis-md-san-francisco/biz/cesar-vega-m-d-northridge/biz/kayekjian-garabed-md-northridge/biz/darush-alan-md-apc-northridge/biz/yamaguchi-michael-md-terra-linda-pediatrics-san-rafael/biz/feldman-stefan-dpm-podtrst-northridge/biz/gabriel-aslanian-dds-md-northridge/biz/kling-michael-od-san-diego/biz/marshall-larry-j-md-lakeside/biz/childrens-primary-care-medical-group-la-jolla-3/biz/marianne-rochester-md-san-diego/biz/sanjay-ghosh-md-faans-san-diego/biz/perry-t-mansfield-md-san-diego/biz/michael-j-o-leary-md-san-diego/biz/karen-e-anderson-dpm-la-mesa/biz/tq-chiropractic-san-diego/biz/stacy-e-hulley-m-d-san-diego/biz/rita-j-feghali-m-d-san-diego/biz/richard-a-kaplan-m-d-san-diego/biz/jason-r-brown-dds-san-diego/biz/lin-cynthia-md-san-diego/biz/thoene-michael-j-md-el-cajon/biz/delois-j-bean-m-d-el-cajon/biz/jockin-yvette-m-md-san-diego/biz/kleid-jack-j-md-san-diego/biz/daniel-y-lee-md-san-diego/biz/richard-j-snyder-md-san-diego/biz/mirkarimi-morteza-san-diego-2/biz/milder-david-g-dds-md-san-diego-2/biz/boone-gary-md-san-diego/biz/robert-maywood-md-san-diego/biz/yorobe-edwin-m-md-san-diego/biz/robert-b-jacob-dds-san-diego/biz/juma-saad-md-encinitas/biz/david-kupfer-md-facs-san-diego/biz/paul-neustein-md-poway/biz/bridge-stephen-s-md-san-diego/biz/jeffrey-h-dysart-md-san-diego/biz/vidush-p-athyal-m-d-san-diego/biz/marc-k-rubenzik-m-d-san-diego/biz/eric-macy-m-d-san-diego/biz/grant-b-neifeld-m-d-san-diego/biz/patrick-c-watson-d-o-san-diego-2/biz/laura-a-mcmillan-m-d-san-diego/biz/ulrika-b-jansson-schumacher-m-d-san-diego/biz/sanjay-dhir-dds-san-diego/biz/sajben-nancy-l-md-la-jolla/biz/marc-e-kramer-md-san-diego/biz/bakst-isaac-md-san-diego/biz/jody-corey-bloom-md-la-jolla/biz/wendy-m-buchi-md-san-diego/biz/valerie-v-gafori-md-san-diego/biz/benito-villanueva-md-san-diego/biz/ronald-j-edelson-md-san-diego/biz/james-a-davis-md-san-francisco-2/biz/lam-chittaphong-dds-san-diego/biz/kj-ben-kim-dds-san-diego/biz/george-madany-md-san-diego/biz/james-p-tasto-dds-san-diego/biz/seibert-chiropractic-poway/biz/rivera-marcelo-r-md-int-med-poway'

review_main_text_list = []
## trian paragraph by paragraph.
#selected_doctors = doctors.loc[0]
#for j in range(0, len(doctors)):
#    item = doctors.loc[j]
#    if (str(item["specialty"]).find("Internal Medicine") != -1 or str(item["specialty"]).find("Family Medicine") != -1):
#        selected_doctors.append(item)
#        
#for i in range(0, len(reviews)):
#    main_text = reviews.loc[i]["review main text"]
#    yelp_url = reviews.loc[i]["page url"].split("?")[0]
#    keep = False
#    for j in range(0, len(doctors)):
#        item = doctors.loc[j]
#        if str(item["yelp"]).find(yelp_url) != -1:
#            if (str(item["specialty"]).find("Internal Medicine") != -1 or str(item["specialty"]).find("Family Medicine") != -1):
#                keep = True
#                break
#    print(i/11694)
#    if keep:
#        paragraphs = main_text.split("\n")
#        for paragraph in paragraphs:
#            if paragraph != "" and len(paragraph) > 300:
#                review_main_text_list.append(paragraph)

for i in range(0, len(reviews)):
    paragraphs = reviews.loc[i]["review main text"]
    review_main_text_list.append(paragraphs)
                
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
cleaned_up_review_list = []
for document in review_main_text_list:
    # Tokenization
    raw = document.lower()
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
    
    cleaned_up_review_list.append(cleaned_text)


'''
!! STEP 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# Find words frequency.
all_words = []
for i in cleaned_up_review_list:
    for j in i:
        all_words.append(j)
fdist = nltk.FreqDist(all_words)
print(fdist.most_common())

words_to_ignore = ["dr", "doctor", "yelp", "dc", "dos", "dmd", "do", "dpm", "mbbch", "md", "od", "rpt", "california", "stanford", "ucla", "ucsf", "usc", "ucsd", "san", "francisco", "diego", "los", "angeles", "oakland", "beverly", "hills", "daly", "santa", "monica", "alamo", "solana", "beach", "poway", "del", "mar", "la", "jolla", "santee", "northridge", "rafael", "panorama", "canada", "flintridge", "glendale", "canyon", "westlake", "village", "riverside", "van", "nuys", "burbank", "tarzana", "encino", "oaks", "fernando", "pasadena", "rancho", "cucamonga", "hollywood", "institute", "york", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "korean", "filipino", "chinese", "embarcadero", "sacramento", "deborah", "irish", "http", "bob", "sculli", "claud", "washington", "meister", "eydelman"]
words_to_ignore = ["dr", "doctor"]
words_to_ignore2 = ["acn","agoura","botox","canal","cardiologist","cedar","colonoscopi","dental","dentist","dermatologist","ear","eczema","eye","face","foot","fungal","hair","haircut","hip","http","jewish","knee","laser","lasik","mr","optometrist","orthoped","pediatrician","perm","physic","physician","salon","straight","surgeon","surgeri","www"]
doctor_names = ['Ganata', 'Brian', 'Stein', 'Jack', 'Mitchell', 'Masami', 'Belcher', 'Rosanelli', 'Weissman', 'Nobel', 'Johnson', 'Michele', 'Shirazi', 'Francis', 'John', 'Vidush', 'Scott', 'Grogan', 'Sands', 'Navneet', 'Abrishami', 'Yun', 'Trott', 'Line', 'Sumati', 'Binstock', 'Ulrika', 'Yousefi', 'Ma', 'Guy', 'Mullens', 'Green', 'Kelly', 'Schiller', 'Ullman', 'Eric', 'Podolin', 'Vartany', 'Sanaz', 'Kashani', 'Collins', 'Jose', 'Mehdizadeh', 'Tiller', 'Baron', 'Maryam', 'Hill', 'Flynn', 'Cecilio', 'Kishibay', 'Stephane', 'Felicia', 'Enayati', 'M', 'Caroline', 'Fahimian', 'Glenn', 'Richard', 'Lam', 'Bakshian', 'Besser', 'Meth', 'Nassos', 'Walker', 'Pena-Robles', 'Forouzesh', 'Rochester', 'Friedman', 'Tri', 'Robert', 'Parnaz', 'Seibert', 'Sterman', 'Hakimian', 'Anupam', 'Rivera', 'Ardizzone', 'Chun', 'Vu', 'Eduardo', 'Lindsay', 'Dave', 'Mirabadi', 'Benbow', 'Fidel', 'Akizuki', 'Carlos C.', 'Wong', 'Rudy', 'Agnes', 'Venuturupalli', 'Fox', 'Urusova', 'Kourosh', 'Potruch', 'Jane', 'Jerry', 'Veronique', 'Hariri', 'Griffiths', 'Wadhwa', 'Maria', 'Stone', 'Lofthus', 'Metaxas', 'Cho', 'Fitzgerald', 'Larry', 'Ellie', 'Armine', 'Virginia', 'Kapla', 'Cohen', 'Yussef', 'Buoncristiani', 'Miao', 'Neidlinger', 'Sajben', 'Gladstein', 'Hillary', 'Samuel', 'Loraine', 'Hakop', 'Benito', 'Mikus', 'Mark', 'Inna', 'Pham', 'Luftman', 'Jacobitz', 'Fine', 'Yuo', 'Annette', 'Miller', 'Shadan', 'Bokosky', 'Sarosy', 'Kathleen', 'Lawrence', 'Boodaghians', 'Tessler', 'Jiu', 'Sandy', 'Babak', 'Michel', 'Yampolsky', 'Roykh', 'Carl', 'Berty', 'Liu', 'Tao', 'Stacy', 'Shahab', 'Catherine', 'Samimi', 'Crane', 'Violet', 'Cynthia', 'Smith', 'Schumacher', 'Kaufman', 'Carlton', 'Friedland', 'Belaga', 'Pourrahimi', 'Montgomery', 'Oo', 'Etie', 'Lakshmi', 'Henderson', 'Shaun', 'Chou', 'Antigone', 'Woods', 'Redlin', 'Yang', 'Carey', 'Pack', 'Sirius', 'Huh', 'Tamara', 'Leon', 'Deleaver-Russell', 'Neifeld', 'Katz', 'Heidenfelder', 'Melanie', 'Mansour', 'Stephen', 'Stern', 'Buchi', 'Yvonne', 'Henry', 'Aurasteh', 'Cafaro', 'Dillingham', 'Obrien', 'Putnam', 'Hyver', 'Ghada', 'Burstein', 'Winchell', 'Peggy', 'Kyoko', 'Garcia', 'Morganroth', 'Shankar', 'Jonah', 'Yorobe', 'Donald', 'Chow', 'Cepkinian', 'Hamidi', 'Chittaphong', 'Snibbe', 'Gennady', 'Tim', 'Joie', 'Rosenbaum', 'Lamont', 'Margossian', 'Silverman', 'Hoosik', 'Lewis', 'Le', 'Justine', 'Linda', 'Maureen', 'Dadvand', 'Mirkarimi', 'Quintero', 'Madany', 'Apelian', 'Ken', 'Berman', 'Corey Bloom', 'Kutzscher', 'Gafori', 'Gevorkyan', 'Akagi', 'Markison', 'Yabumoto', 'Valenton', 'Amir', 'Degolia', 'Kruse', 'Davidson', 'Ahluwalia', 'George', 'Elisa', 'Stephanie', 'Babajanian', 'Schultz', 'Sonal', 'Mohammad', 'Thoene', 'Wan', 'Babapour', 'Richards', 'Minkowsky', 'Galitzer', 'Myunghae', 'Vadim', 'Hess', 'Pegah', 'Kawashiri', 'Jay', 'Kram', 'Aguilar', 'Paiement', 'Theodore', 'Oleary', 'Izumi', 'Kathryn', 'Robin', 'Edgardo', 'Yoon', 'Ashish', 'Mike', 'Farinoush', 'Kimberly', 'Hornbrook', 'Betsy', 'Isaac', 'Massoudi', 'Gottlieb', 'Falcon', 'Athyal', 'Chu', 'Mynsberge', 'Helen', 'Collin', 'Ambartsumyan', 'Stacey', 'Romeo', 'Goei', 'Colbert', 'Hong', 'Brar', 'Mario', 'Levin', 'Charles', 'Fischbein', 'Meghan', 'Navab', 'Luis', 'Xiang', 'Orecklin', 'Banooni', 'Girard', 'Abe', 'Bhoot', 'Ebrahimi', 'Saliman', 'Joseph', 'Koltzova-Rang', 'Fakhouri', 'Matt', 'Dhir', 'Ghosh', 'Walter', 'Mehras', 'Pudberry', 'Satinder', 'Mehranpour', 'Rajan', 'Mann', 'Chang', 'Juliana', 'Sid', 'Hernandez', 'Arretz', 'Gregory', 'Marion', 'Howell', 'Greg', 'Van', 'Akhavan', 'Kerry', 'Blechman', 'Cha', 'Leopoldo', 'Jeannie', 'Cook', 'Giuliano', 'Scheinhorn', 'Weber', 'Raymond', 'Kupfer', 'Leung', 'Sarvenaz', 'Tuft', 'Amanda', 'Chobanian', 'Sima', 'Marc', 'Susan', 'Minoo', 'Ali', 'Eubanks', 'Soliman', 'Diego', 'Galpin', 'Sandler', 'Cinque', 'Eastman', 'Low', 'Marino', 'Lily', 'Barzman', 'Anita', 'Melissa', 'Maloney', 'Davies', 'Khoubnazar', 'Heather', 'Rachel', 'Molayem', 'Nita', 'Dan', 'Alison', 'Anderson', 'Alexander', 'Juma', 'Pool', 'Gin', 'Ganjianpour', 'Foster', 'Marriott', 'Bakst', 'Colette', 'Rust', 'Rahimian', 'Malamed', 'Jones', 'Nancy', 'Perl', 'Chester', 'Jasmine', 'Jared', 'Ehmer', 'Kalpari', 'Nazareth', 'Cortland', 'Chase', 'Arash', 'Birns', 'Bill', 'Eugene', 'Everson', 'David', 'Berkowitz', 'Nguyentu', 'Morteza', 'Zareh', 'May', 'Christine', 'Abuav', 'Amanuel', 'Mahnaz', 'Gingold', 'Desiree', 'Bruce', 'Parvin', 'McLucas', 'Azer', 'Kassab', 'Mandel', 'Shamie', 'Panosian', 'Andrea', 'Jerome', 'Sanjay', 'Bhuva', 'Gayane', 'Baker', 'Asaf', 'Gaminchi', 'Tran', 'Mamta', 'Philippe', 'Tang', 'Peiman', 'Rosen', 'Shiell', 'Iskander', 'Stefan', 'Levy', 'Jessica', 'Fung', 'Bailey', 'Edmund', 'Borah', 'Hulley', 'Armando', 'Kambiz', 'Stuart', 'Youssef', 'Blumstein', 'Groth', 'Frederica', 'Karpman', 'Palm', 'Nguyen', 'Davis', 'Zapata', 'Aizuss', 'William', 'Amirtharajah', 'Darush', 'Judy', 'Dina', 'Hans', 'Andre', 'Gary', 'Regan', 'Christman', 'Riedler', 'Justus', 'Yulia', 'Allan', 'Miclau', 'Edwin', 'Scalise', 'Bardowell', 'Jeanne', 'Reid', 'Agata', 'Reiche', 'Laurie', 'Castillo', 'Abraham', 'Arnold', 'Levi', 'Pedro', 'Thomas', 'Paul', 'Boska', 'Doan', 'Ng', 'Moradzadeh', 'Wolfe', 'Khalil', 'Park', 'Liao', 'Timothy', 'Nikolaj', 'Sinclair', 'Karlsberg', 'Yelding-Sloan', 'Marcelo', 'Graham', 'Shih', 'Bhatia', 'Waleed', 'Foxman', 'Khosravi', 'Yee', 'Inouye', 'Joshua', 'Edelson', 'Najarian', 'Klasky', 'Knox', 'Vaughn', 'Boykoff', 'Teguh', 'Kwang', 'Oliver', 'Alex', 'Debra', 'Snunit', 'Irene', 'Auerbach', 'Kearney', 'Gupta', 'Royeen', 'Tristan', 'Binder', 'Min', 'Kevin', 'Monali', 'Golshani', 'Emrani', 'Rostker', 'Lieu', 'Lepor', 'Massry', 'Claire', 'Rita', 'Cotter', 'Drell', 'Zdzislaus', 'Schechter', 'Vail', 'Kind', 'Marilyn', 'Jockin', 'Mantell', 'Kramer', 'Shaden', 'Sherman', 'Rafael', 'Holly', 'Jick', 'Uyeki', 'Ginsberg', 'Gold', 'Afshine', 'Casper', 'Khin', 'Hofstadter', 'Safvati', 'Suhail', 'Nick', 'Krames', 'Afshin', 'Myers', 'Sawsan', 'Meng', 'Mayer', 'Silberstein', 'Barkley', 'Kamanine', 'Warren', 'Shindy', 'Aslanian', 'Dembo-Smeaton', 'Prathipati', 'Rosenberg', 'Jeremy', 'Cepeda', 'Jade', 'Barnhard', 'Duong', 'Habib', 'Eisenhart', 'Tamayo', 'Goodman', 'Danelle', 'Christopher', 'Darakjian', 'Hal', 'Marquez', 'Moshe', 'Harry', 'Philip', 'Shirley', 'Alen', 'Hsu', 'Tu', 'Renee', 'Kiriakos', 'Estes', 'Kayekjian', 'Lynn', 'Kim', 'Perry', 'Tachdjian', 'Marnell', 'Karen', 'Sosa', 'Vahan', 'Talreja', 'Estwick', 'Floyd', 'Nilesh', 'Adrien', 'Yamaguchi', 'Tamer', 'Oechsel', 'Sandeep', 'Tabsh', 'Bowden', 'Bohn', 'Day', 'Edward', 'James', 'Zonia', 'Mani', 'Young', 'Stamper', 'Simon', 'Dougherty', 'Touradge', 'Otoole', 'Bresnick', 'Quan', 'Leslie', 'Sellman', 'Matthew', 'Donna', 'Swerdlow', 'Daphne', 'Mohamed', 'Pigeon', 'Marshall', 'Sawusch', 'Newcomer', 'Ganelis', 'Dysart', 'Hamilton', 'Russo', 'Lavi', 'Hayward', 'Cheryl', 'Jessely', 'Rabinovich', 'Lee', 'Mohammed', 'Rashti', 'Sarkisian', 'Factor', 'Jeannine', 'Laurence', 'Skoulas', 'Amarpreet', 'Bajaj', 'Beller', 'Villanueva', 'Lenzkes', 'Hu', 'Victor', 'Blair', 'Tarick', 'Alla', 'Dardashti', 'Tsoi', 'Sumeer', 'Mancherian', 'Levinson', 'Borookhim', 'Shervin', 'Flores', 'Bessie', 'Irina', 'Levine', 'Schofferman', 'Draupadi', 'Lipton', 'Epstein', 'Yen', 'Agbuya', 'Ruder', 'Mansfield', 'Rawat', 'Lin', 'Alessi', 'Eshaghian', 'Takahashi', 'Keyvan', 'Rashtian', 'Derrick', 'Arya Nick', 'Bert', 'Quock', 'Shukri', 'Flaherty', 'Kvitash', 'Man', 'Diamond', 'Choi', 'Lisa', 'Miremadi', 'Wendy', 'Liau', 'Lofquist', 'Shunpei', 'Douglas', 'Saito', 'Roya', 'Jenkin', 'Ralph', 'Dicks', 'Moghissi', 'De Luna', 'Michael', 'Nadiv', 'Schwanke', 'Kapoor', 'Starrett', 'Yip', 'Akash', 'Reese', 'Vera', 'Bridge', 'Feghali', 'Rosenbach', 'Jennifer', 'Brenda', 'Garg', 'Haleh', 'Chin', 'Yoo', 'Harold', 'Goodwin', 'Feltman', 'Shehnaz', 'Wu', 'Hendry', 'Emmanuel', 'Elena', 'Lakshman', 'Danzer', 'Maurice', 'Farnaz', 'Rose', 'Leitner', 'Khodabakhsh', 'Nam', 'Simoni', 'Parviz', 'Biderman', 'Snyder', 'Jacob', 'Anmar', 'Justin', 'Soleimani', 'Waring', 'Mueller', 'Fishman', 'Custis', 'Ann', 'Gilman', 'Nunes', 'Flach', 'Gores', 'Larian', 'Dana', 'Yokoyama', 'Dalwani', 'Chunbong', 'Grady', 'Carlos', 'Diana', 'Nora', 'Roberts', 'Elayne', 'Yvette', 'Weiss', 'Larisse', 'McMillan', 'Cesar', 'Kang', 'Chan', 'Gabriel', 'Rabin', 'Milder', 'Chenette', 'Lawton', 'Garabed', 'Malhotra', 'Char', 'Makassebi', 'Patel', 'Mesler', 'Eisele', 'Kenneth', 'Tamkin', 'Salit', 'Abhay', 'Keith', 'Sternberg', 'Wolff', 'Cortez', 'Rhee', 'Plance', 'Vincent', 'Pivo', 'Boone', 'Jonathan', 'Rosanna', 'Sangdo', 'Suzanne', 'Yu', 'Vega', 'Strom', 'Lau', 'Ben-Ozer', 'Hoyman', 'Bryan', 'Garbis', 'Hattori', 'Kahn-Rose', 'Macy', 'Woolf', 'Tamarin', 'Genen', 'Gaytan', 'Ramtin', 'Valerie', 'Atkin', 'Solomon', 'Fossett', 'Mahshid', 'Alikpala', 'Neustein', 'Tasto', 'Arjang', 'Ebrahim', 'Lief', 'Lara', 'Raul', 'Leah', 'Raffi', 'Darragh', 'Howard', 'Pedrotty', 'Serena', 'Chandrasekhar', 'Irving', 'Berdjis', 'Brown', 'Diggs', 'Sverdlov', 'Reza', 'Alza', 'Felipe', 'Yamada', 'Frederick', 'Jerrold', 'Orpilla', 'Peter', 'Barry', 'Sakhai', 'Alan', 'Kadner', 'Patrick', 'Jeffrey', 'Armstrong', 'Kleid', 'Paula', 'Tahani', 'Garber', 'Watson', 'Melvin', 'Gustavo', 'Roth', 'Smaili', 'Hoang', 'Devron', 'Daws', 'Tuan', 'Trojnar', 'Bong', 'Katherine', 'Nesari', 'Kawilarang', 'Ronald', 'Kamran', 'Gordon', 'Menendez', 'Bortz', 'Massey', 'Rubenzik', 'Alfred', 'Marina', 'Fawaz', 'Shafipour', 'Bloomfield', 'Feldman', 'Chua', 'Pouya', 'Peyman', 'Norman', 'Stefani', 'Yazdani', 'Sameer', 'Dohad', 'Kurtz', 'Molato', 'Refoa', 'Marie', 'Engel', 'Pamela', 'Caroll', 'Daneshgar', 'Sun', 'Haley', 'Valentina', 'Leonard', 'Maeck', 'Michelle', 'Roohipour', 'Faisal', 'Payam', 'Kramar', 'Kerman', 'Sherwin', 'Khoury', 'Garrick', 'Leyli', 'Wanski', 'Cardon', 'Pean', 'Assil', 'Bahadori', 'Andrew', 'Rodney', 'Chiu', 'Taaly', 'Remy', 'Fisher', 'Sharon', 'Melody', 'Bala', 'Armen', 'Mobasser', 'Joy', 'Nader', 'Beeve', 'Hammond', 'Vanhale', 'Cheung', 'Cheng', 'Kranson', 'Sloan', 'Delois', 'Silani', 'Wieder', 'Vafaie', 'Chien', 'Brandeis', 'Su', 'Gamache', 'Ray', 'Bean', 'Mohana', 'Guido', 'Starnes', 'Chong', 'Martin', 'Daniel', 'Biana', 'Schulman', 'Marianne', 'Randolph', 'Morris', 'Xilin', 'Bailony', 'Reynaldo', 'Caldwell', 'Song', 'Herbert', 'Saad', 'Elgan', 'Esther', 'Nasimeh', 'Paz', 'Greenberg', 'Hopper', 'Derek', 'Grant', 'Vlad', 'Kaplan', 'Amini', 'Albert', 'Kling', 'Benjamin', 'Sam', 'Cabrera', 'Chiu-Collins', 'Wolfson', 'Margo', 'Cowan', 'Chen', 'Payman', 'Rish', 'Sanders', 'Cameron', 'Owens', 'Phillips', 'Dao', 'Allison', 'Maywood', 'Elliot', 'Jody', 'Thaik', 'Korchek', 'Eng', 'Ton', 'Thuc', 'Nathan', 'Glen', 'Bickman', 'Reyes', 'Sarafzadeh', 'Hansen', 'Yuan', 'Nikole', 'Mervin', 'Aiache', 'Iwata', 'Considine', 'Tyler', 'Rodriguez', 'Dawn', 'Steven', 'Carol', 'E.', 'Kamyar', 'Tin', 'Jason', 'Moy', 'Duncan', 'Merilynn', 'Dye', 'Chaves', 'Sajedi', 'Strelkoff', 'Lattanza', 'Janet', 'Joan', 'Elizabeth', 'Weller', 'Swamy', 'Rupsa', 'Laura', 'Shu', 'Joana', 'Jan', 'Joel', 'Rubinstein', 'Co']
doctor_names_cleaned_up = []
for name in doctor_names:
    doctor_names_cleaned_up.append(p_stemmer.stem(name.lower()))

words_to_ignore += doctor_names_cleaned_up

# Filter out the words that we need to ignore.
cleaned_up_review_list2 = list(cleaned_up_review_list)
for tokens in cleaned_up_review_list2:    
    # Delete specificed words.
    j = 0
    while j < len(tokens):
        if tokens[j] in words_to_ignore:
            del tokens[j]
        else:
            j += 1
    
# Generate corpus.
dictionary = gensim.corpora.Dictionary(cleaned_up_review_list2)
print(dictionary.token2id)  
corpus = [dictionary.doc2bow(word) for word in cleaned_up_review_list2]

# TF-IDF
#tfidf = models.TfidfModel(corpus)
#corpus = tfidf[corpus]

# Setting up LDA model.
no_of_topics = 15
passes_in = 100

# Training LDA model and checking results.
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_of_topics, id2word = dictionary, passes=passes_in, alpha='asymmetric')
pickle.dump(ldamodel, open("/home/nwang/proj/data/lda_model/ldamodel_v5.pickle", "wb"))
pickle.dump(dictionary, open("/home/nwang/proj/data/lda_model/dictionary_v5.pickle", "wb"))
pickle.dump(corpus, open("/home/nwang/proj/data/lda_model/corpus_v5.pickle", "wb"))

# Check topics.
topic_list = ldamodel.print_topics(num_topics=no_of_topics, num_words=15)
for index, i in enumerate(topic_list):
    str1 = str(i[1])
    for c in "0123456789+*\".":
        str1 = str1.replace(c, "")
    str1 = str1.replace("  ", " ")
    print(str1)

###############################################################################
###############################################################################
# Fake corpus test.
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


for index, i in enumerate(topic_list_1):
    str1 = str(i[1])
    print(str1)
print()
for index, i in enumerate(topic_list_2):
    str1 = str(i[1])
    print(str1)
