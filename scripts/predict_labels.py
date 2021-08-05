'''
COMMAND LINE EXAMPLE:
python3 predict_labels.py 02 ../N2C2/new_data_txt 1
'''

################
### packages ###
################

# for reading files
import pandas as pd 
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
import re
import argparse
import textwrap
import pickle

# for cleaning discharge summaries
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import warnings

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from nltk.tokenize import RegexpTokenizer
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

##############
### labels ###
##############
MATCHING =  1
NOT_MATCHING = 0
ABSTAIN = -1

##############
### models ###
##############
sys.path.insert(1, '../models')
from lf_models import lf_model_svm_ade_only_linear
from lf_models import lf_model_svm_ade_only_polynomial
from lf_models import lf_model_svm_ade_only_sigmoid
from lf_models import lf_model_svm_ade_only_rbf
from lf_models import lf_model_svm_ade_only_gridsearch
from lf_models import lf_model_svm_ade_drug_linear
from lf_models import lf_model_svm_ade_drug_polynomial
from lf_models import lf_model_svm_ade_drug_sigmoid
from lf_models import lf_model_svm_ade_drug_rbf
from lf_models import lf_model_svm_ade_drug_gridsearch
from lf_models import lf_model_svm_reason_drug_linear
from lf_models import lf_model_svm_reason_drug_polynomial
from lf_models import lf_model_svm_reason_drug_sigmoid
from lf_models import lf_model_svm_reason_drug_rbf
from lf_models import lf_model_svm_reason_drug_gridsearch
from lf_models import lf_model_multi_label_one_vs_rest
from lf_models import lf_model_multi_label_binary_relevance
from lf_models import lf_model_multi_label_classifier_chains
from lf_models import lf_model_multi_label_label_powerset
from lf_models import lf_model_multi_label_adapted_algorithm
from lf_models import lf_model_flair

#############
### rules ###
#############
sys.path.insert(1, '../rules')
from rule1 import lf_ade_drug_single
from rule2 import lf_ade_drug_pair
from rule3 import lf_ade_drug_pair_lem
from rule4 import lf_ade_drug_pair_lem_keyword_triggers
from rule5 import lf_sider2_triggers
from rule6 import lf_sider2_triggers_25words
from rule7 import lf_semmeddb_triggers
from rule8 import lf_keyword_triggers
from rule9 import lf_paper_triggers
from rule10 import lf_paper_triggers_200char
from rule11 import lf_paper_triggers_200char_negate
from rule12 import lf_paper_triggers_25words

########################
### models and rules ###
########################

models_dict = {
    "1": lf_model_svm_ade_only_linear,
    "2": lf_model_svm_ade_only_polynomial,
    "3": lf_model_svm_ade_only_sigmoid,
    "4": lf_model_svm_ade_only_rbf,
    "5": lf_model_svm_ade_only_gridsearch,
    "6": lf_model_svm_ade_drug_linear,
    "7": lf_model_svm_ade_drug_polynomial,
    "8": lf_model_svm_ade_drug_sigmoid,
    "9": lf_model_svm_ade_drug_rbf,
    "10": lf_model_svm_ade_drug_gridsearch,
    "11": lf_model_svm_reason_drug_linear,
    "12": lf_model_svm_reason_drug_polynomial,
    "13": lf_model_svm_reason_drug_sigmoid,
    "14": lf_model_svm_reason_drug_rbf,
    "15": lf_model_svm_reason_drug_gridsearch,
    "16": lf_model_multi_label_one_vs_rest,
    "17": lf_model_multi_label_binary_relevance,
    "18": lf_model_multi_label_classifier_chains,
    "19": lf_model_multi_label_label_powerset,
    "20": lf_model_multi_label_adapted_algorithm,
    "21": lf_model_flair
}

models_dict_desc = {
    "0": "No Models Selected",
    "1": "ADE-Only Prediction using Linear SVM",
    "2": "ADE-Only Prediction using Polynomial SVM",
    "3": "ADE-Only Prediction using Sigmoid SVM",
    "4": "ADE-Only Prediction using RBF SVM",
    "5": "ADE-Only Prediction using GridSearch SVM",
    "6": "ADE-Drug Prediction using Linear SVM",
    "7": "ADE-Drug Prediction using Polynomial SVM",
    "8": "ADE-Drug Prediction using Sigmoid SVM",
    "9": "ADE-Drug Prediction using RBF SVM",
    "10": "ADE-Drug Prediction using GridSearch SVM",
    "11": "Reason-Drug Prediction using Linear SVM",
    "12": "Reason-Drug Prediction using Polynomial SVM",
    "13": "Reason-Drug Prediction using Sigmoid SVM",
    "14": "Reason-Drug Prediction using RBF SVM",
    "15": "Reason-Drug Prediction using GridSearch SVM",
    "16": "Multi-Label Prediction using One vs Rest",
    "17": "Multi-Label Prediction using Binary Relevance",
    "18": "Multi-Label Prediction using Classifier Chains",
    "19": "Multi-Label Prediction using Label Powerset",
    "20": "Multi-Label Prediction using Adapted Algorithm",
    "21": "Flair"
}

rules_dict = {
    "1": lf_ade_drug_single,
    "2": lf_ade_drug_pair,
    "3": lf_ade_drug_pair_lem,
    "4": lf_ade_drug_pair_lem_keyword_triggers,
    "5": lf_sider2_triggers, 
    "6": lf_sider2_triggers_25words,
    "7": lf_semmeddb_triggers,
    "8": lf_keyword_triggers,
    "9": lf_paper_triggers,
    "10": lf_paper_triggers_200char,
    "11": lf_paper_triggers_200char_negate,
    "12": lf_paper_triggers_25words
}

rules_dict_desc = {
    "0": "No Rules Selected",
    "1": "lf_ade_drug_single - any keywords in ade_drug_single found in discharge summary",
    "2": "lf_ade_drug_pair - any pair of keywords in ade_drug_pair found in discharge summary",
    "3": "lf_ade_drug_pair_lem - any pair of lemmatised keywords in ade_drug_pair found in discharge summary",
    "4": "lf_ade_drug_pair_lem_keyword_triggers - any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary",
    "5": "lf_sider2_triggers - any pair of trigger words in sider2_triggers found in discharge summary",
    "6": "lf_sider2_triggers_25words - any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary",
    "7": "lf_semmeddb_triggers - any pair of trigger words in semmeddb_triggers found in discharge summary",
    "8": "lf_keyword_triggers - any trigger word in keyword_triggers found in discharge summary",
    "9": "lf_paper_triggers - any trigger word in paper_triggers found in discharge summary",
    "10": "lf_paper_triggers_200char - any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary",
    "11": "lf_paper_triggers_200char_negate - any trigger word in paper_triggers within 200 characters of any keyword in negate found in discharge summary",
    "12": "lf_paper_triggers_25words - any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary"
}

#################
### argparser ###
#################

### create the parser ###
parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter)

### add the arguments ###
# folder
parser.add_argument('folder_number', type=str, 
    help=''' 
        folder number that has the label model and majority label voter to be used to predict labels
        ''')

# data
parser.add_argument('data_path', type=str,
    help='''
        data path of folder with discharge summaries to have labels predicted
        ''')

# model
parser.add_argument('model', type=str, 
    help=''' 
        1. Label Model
        2. Majority Label Voter
        ''')

### execute the parse_args() method ###
args = parser.parse_args()

folder_number = args.folder_number
data_path = args.data_path
model_label = args.model

# load model for prediction
model_path = "../outputs/run" + folder_number
print('MODEL PATH', model_path)
if model_label == "1" :
    model_name = 'label_model.pkl'
elif model_label == "2" :
    model_name = 'majority_label_voter.pkl'
with open(model_path + "/" + model_name, 'rb') as file :
    model = pickle.load(file)

#################
###  info.csv ###
#################

info = pd.read_csv(model_path + "/info.csv")
models = info['models'][0]
rules = info['rules'][0]
if type(models) == str :
    models_list = models.split(",")
else :  
    models_list = [str(models)]
if type(rules) == str :
    rules_list = rules.split(",")
else :
    rules_list = [str(rules)]

lfs= []

### print ###
print("\n")
print("output folder path:")
print(model_path)
print("\n")

# get models #
print("models selected:")
for i in range(0, len(models_list)) :
    print(models_list[i], ":", models_dict_desc[models_list[i]])
    if models_list[i] != "0" :
        lfs.append(models_dict[models_list[i]])

print("\n")
print("rules selected:")
# get rules #
for i in range(0, len(rules_list)) :
    print(rules_list[i], ":", rules_dict_desc[rules_list[i]])
    if rules_list[i] != "0" :
        lfs.append(rules_dict[rules_list[i]])

###########################
### discharge summaries ###
###########################

# clean discharge summaries
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

# remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
stemmer = SnowballStemmer("english")

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# get new data
new_data_patient = []
new_data_summary = []
for file in os.listdir(data_path):
    if file != ".DS_Store" :
        with open(join(data_path, file), 'r') as document_summary_file:
            data = " ".join(line.strip() for line in document_summary_file)
            new_data_patient.append(file[:-4])
            new_data_summary.append(data)
new_data_df = pd.DataFrame({"patient": new_data_patient, "summary": new_data_summary})

# clean new_data_df discharge summaries
new_data_df_cleaned = new_data_df.copy()
new_data_df_cleaned['cleaned_summary'] = new_data_df_cleaned['summary']
for i in range(0, len(new_data_df_cleaned)) :
    new_data_df_cleaned.cleaned_summary[i] = new_data_df_cleaned.cleaned_summary[i].lower()
    new_data_df_cleaned.cleaned_summary[i] = cleanHtml(new_data_df_cleaned.cleaned_summary[i])
    new_data_df_cleaned.cleaned_summary[i] = cleanPunc(new_data_df_cleaned.cleaned_summary[i])
    new_data_df_cleaned.cleaned_summary[i] = keepAlpha(new_data_df_cleaned.cleaned_summary[i])
    new_data_df_cleaned.cleaned_summary[i] = removeStopWords(new_data_df_cleaned.cleaned_summary[i])
    # new_data_df_cleaned.cleaned_summary[i] = stemming(new_data_df_cleaned.cleaned_summary[i])

print("\n")
print("new dataset cleaned")

###################
###  LF applier ###
###################

applier = PandasLFApplier(lfs=lfs) 
L_new_data = applier.apply(df=new_data_df_cleaned)

#################
### prediction ###
#################

new_labels = model.predict(L_new_data)  
all_prob = model.predict_proba(L_new_data)
new_prob = []
for i in range(0, len(new_labels)) :
    if new_labels[i] == 0 :
        new_prob.append(all_prob[i][0])
    elif new_labels[i] == 1 :
        new_prob.append(all_prob[i][1])
print("predicted labels generated")
# add prediction to new_data_df    
new_data_df['predicted_label'] = new_labels
new_data_df['predicted_probability'] = new_prob
# save as .csv   
file_name = model_path + "/" + str(models) + "_" + str(rules) + "_" + model_name[:-4] + ".csv"
new_data_df.to_csv(file_name, index=False)
print("predicted labels saved as " + file_name)
print("\n")
print("###########")
print("### END ###")
print("###########")
print("\n")
