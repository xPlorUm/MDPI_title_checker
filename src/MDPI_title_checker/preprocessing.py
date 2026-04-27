
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def preprocess(texts):
    return [normalize(text) for text in texts]
    
# Normalization does break semantic comparison quite often, hence just skip it
# 

# def normalize(string):
#     # convert to lower case
#     no_number_string = string.lower()
    
#     # remove punctuations
#     no_punc_string = re.sub(r'[^\w\s]', '', no_number_string)
    
#     # remove whitespace
#     no_wspace_string = no_punc_string.strip()
    
#     # remove stopwords
#     lst_string = [no_wspace_string][0].split()
#     no_stpwords_string = ""
#     for i in lst_string:
#         if not i in stop_words:
#             no_stpwords_string += i + ' '
#     no_stpwords_string = no_stpwords_string[:-1] # remove last space
    
#     return no_stpwords_string

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()