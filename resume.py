#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
import requests
import constants as cs
import pandas as pd
import streamlit as st
from io import BytesIO
import hydralit_components as hc
import docx2txt
import pdfplumber
import pickle
from pickle import load
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot  as plt
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
stop=set(stopwords.words('english'))

sys.coinit_flags = 0
# load pre-trained model
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import Matcher
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

mfile = BytesIO(requests.get('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/RF.pkl?raw=true').content)
model = load(mfile)

mfile1 = BytesIO(requests.get('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/vectorizer.pkl?raw=true').content)
model1 = load(mfile1)

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

#specify the primary menu-defination
menu_data = [
    {'icon':"fa fa-address-card",'label':'Resume Classification'},
    {'icon':"far fa-file-word",'label':'Resume Parser'},
]
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name = 'Home',
    login_name = None,
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

if menu_id == 'Home':
    
    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.001)
       my_bar.progress(percent_complete + 1)
    tab1,tab2 = st.tabs(["📝 About Project","📉 Data Overview"])
    with tab1:
        st.title('About Project')
        st.subheader('Resume Classificaiton')

        st.header(f"Business Objective: The document classification solution should significantly reduce the manual human effort in the HRM. It should achieve a higher level of accuracy and automation with minimal human intervention.")
        st.image("https://github.com/Siddhant1803/Resume_Classification/blob/main/Image/Designer.png?raw=true",width=400)
        st.markdown("### **Abstract:**\n\
#### A resume is a brief summary of your skills and experience. Companies recruiters and HR teams have a tough time scanning thousands of qualified resumes. Spending too many labor hours segregating candidates resume's manually is a waste of a company's time, money, and productivity. Recruiters, therefore, use resume classification in order to streamline the resume and applicant screening process. NLP technology allows recruiters to electronically gather, store, and organize large quantities of resumes. Once acquired, the resume data can be easily searched through and analyzed.\
\n\
#### Resumes are an ideal example of unstructured data. Since there is no widely accepted resume layout, each resume may have its own style of formatting, different text blocks and different category titles. Building a resume classification and gathering text from it is no easy task as there are so many kinds of layouts of resumes that you could imagine.\n\
\
### Introduction:\n\
\
#### In this project we dive into building a Machine learning model for Resume Classification using Python and basic Natural language processing techniques. We would be using Python's libraries to implement various NLP (natural language processing) techniques like tokenization, lemmatization, parts of speech tagging, etc.\n\
\
#### A resume classification technology needs to be implemented in order to make it easy for the companies to process the huge number of resumes that are received by the organizations. This technology converts an unstructured form of resume data into a structured data format. The resumes received are in the form of documents from which the data needs to be extracted first such that the text can be classified or predicted based on the requirements. A resume classification analyzes resume data and extracts the information into the machine readable output. It helps automatically store, organize, and analyze the resume data to find out the candidate for the particular job position and requirements. This thus helps the organizations eliminate the error-prone and time-consuming process of going through thousands of resumes manually and aids in improving the recruiters’ efficiency.\n\
\
#### The basic data analysis process is performed such as data collection, data cleaning, exploratory data analysis, data visualization, and model building. The dataset consists of two columns, namely, Role Applied and Resume, where ‘role applied’ column is the domain field of the industry and ‘resume’ column consists of the text extracted from the resume document for each domain and industry.\n\
\
#### The aim of this project is achieved by performing the various data analytical methods and using the Machine Learning models and Natural Language Processing which will help in classifying the categories of the resume and building the Resume Classification Model.")

        st.markdown('<img align="right" alt="code"  height="200" width="200" src = "https://static.wixstatic.com/media/15e6c3_8f8cac375de143dc9d1d552090d975cf~mv2.gif">', unsafe_allow_html=True)

    with tab2:
        st.title("Data Overview")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File Type")
            st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/file_types1_2.png?raw=true')

        with col2:
            st.subheader("Classes from Dataframe")
            st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/multiple_profiles1_2.png?raw=true')
            st.write('Data only contain 4 categories')

        st.subheader("Word Count Category Wise")
        st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/peoplesoftwordcloud.png?raw=true')
        st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/sqldeveloperwordcloud.png?raw=true')
        st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/reactdeveloperwordcloud.png?raw=true')
        st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/workdaywordcloud.png?raw=true')

        st.subheader("Over Word Count in All Category")
        st.image('https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Image/worldcloudall.png?raw=true')
                         

def extract_text_from_docx(docx_path):

    if docx_path.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')
    else :
        st.warning(body="Not Supported file Format Found")

resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume        


# In[5]:

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


# Function to extract experience details
def expDetails(Text):
    global sent
   
    Text = Text.split()
   
    for i in range(len(Text)-2):
        Text[i].lower()
        
        if Text[i] ==  'years':
            sent =  Text[i-2] + ' ' + Text[i-1] +' ' + Text[i] +' '+ Text[i+1] +' ' + Text[i+2]
            l = re.findall(r'\d*\.?\d+',sent)
            for i in l:
                a = float(i)
            return(a)
            return (sent)

#Extracting skills
def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
            
    # reading the csv file
    data = pd.read_csv("https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Resume.csv") 
            
    # extract values
    skills = list(data.columns.values)
            
    skillset = []

 # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
            
    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

if menu_id == 'Resume Classification':
    with hc.HyLoader('Please Wait!',hc.Loaders.standard_loaders,index=5):
        time.sleep(0.8)

    st.title("RESUME CLASSIFICATION")
        
    st.subheader('Upload Resumes')

    st.write(r'Note: Classifies only Peoplesoft, Workday, SQL Developer and ReactJS Developer Resumes')
    
    tab1, tab2 = st.tabs(["💾 Single File","📁 Multiple Files"])

    with tab1:
        upload_file1 = st.file_uploader('', type= ['docx'], accept_multiple_files=False)   
        st.write('*Note: For different Resumes Results Reupload')  
        if upload_file1 is not None:
            displayed=extract_text_from_docx(upload_file1)
            cleaned=preprocess(display(upload_file1))
            predicted= model.predict(model1.transform([cleaned]))
            if predicted == 3:
                classify = "Workday"
                st.header("The "+ upload_file1.name +" is Applied for"+ " " + classify + " " + "Profile")
                expander = st.expander("See Resume")
                expander.write(displayed)
                st.image("https://www.workday.com/content/dam/web/en-us/images/social/workday-og-theme.png",width=480)
                
            elif predicted == 2:
                classify = "SQL Developer"
                st.header("The "+ upload_file1.name +" is Applied for"+ " " + classify + " " + "Profile")
                expander = st.expander("See Resume")
                expander.write(displayed)
                st.image("https://wallpaperaccess.com/full/2138094.jpg",width=480)
                
            elif predicted == 1:
                classify = "React Developer"
                st.header("The "+ upload_file1.name +" is Applied for"+ " " + classify + " " + "Profile")
                expander = st.expander("See Resume")
                expander.write(displayed)
                st.image("https://i0.wp.com/www.electrumitsolutions.com/wp-content/uploads/2020/12/wp4923992-react-js-wallpapers.png",width=480)
                
            elif predicted == 0:
                classify = "Peoplesoft"
                st.header("The "+ upload_file1.name +" is Applied for"+ " " + classify + " " + "Profile")
                expander = st.expander("See Resume")
                expander.write(displayed)
                st.image("https://s3.amazonaws.com/questoracle-staging/wordpress/uploads/2019/07/25164143/PeopleSoft-Now.jpg",width=480)
                

    with tab2:
        st.write('Upload Folder Containing Multiple .docx Files')

        file_type=pd.DataFrame(columns=['Uploaded File', 'Experience', 'Skills', 'Predicted Profile'], dtype=object)
        filename = []
        predicted = []
        experience = []
        skills = []

        upload_file2 = st.file_uploader('', type= ['docx'], accept_multiple_files=True)
        
        for doc_file in upload_file2:
            if doc_file is not None:
                filename.append(doc_file.name)   
                cleaned=preprocess(extract_text_from_docx(doc_file))
                predicted.append(model.predict(model.transform([cleaned])))
                extText = extract_text_from_docx(doc_file)
                exp = expDetails(extText)
                experience.append(exp)
                skills.append(extract_skills(extText))

        if len(predicted) > 0:
            file_type['Uploaded File'] = filename
            file_type['Experience'] = experience
            file_type['Skills'] = skills
            file_type['Predicted Profile'] = classify
            # file_type
            st.table(file_type.style.format({'Experience': '{:.1f}'}))

            # opt = st.radio("Choose candidate with prospective of :",["Skills","Experience(years)"])
            # if opt == "Skills":
            #     Skill_option = file_type["Skills"].unique().tolist()
            #     Skill = st.selectbox("Choose the candidate by selecting skills",Skill_option, 0)

if menu_id == 'Resume Parser':

    with hc.HyLoader('Please Wait!',hc.Loaders.standard_loaders,index=5):
        time.sleep(2)
    
    # FOR INDIAN RESUME RUN THE BELOW FUNCTION TO EXTRACT MOBILE NUMBER
    def extract_mobile_number(text):
        phone= re.findall(r'[8-9]{1}[0-9]{9}',text)
        
        if len(phone) > 10:
            return '+' + phone
        else:
            return phone

    def extract_email(text):
        email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None
    # Function to remove punctuation and tokenize the text
    def tokenText(extText):
       
        # Remove punctuation marks
        punc = r'''!()-[]{};:'"\,.<>/?@#$%^&*_~'''
        for ele in extText:
            if ele in punc:
                puncText = extText.replace(ele, "")
                
        # Tokenize the text and remove stop words
        stop_words = set(stopwords.words('english'))
        puncText.split()
        word_tokens = word_tokenize(puncText)
        TokenizedText = [w for w in word_tokens if not w.lower() in stop_words]
        TokenizedText = []
      
        for w in word_tokens:
            if w not in stop_words:
                TokenizedText.append(w)
        return(TokenizedText)

    # Function to extract Name and contact details
    def extract_name(Text):
        name = ''  
        for i in range(0,3):
            name = " ".join([name, Text[i]])
        return(name)
    
    # Grad all general stop words
    STOPWORDS = set(stopwords.words('english'))
    
    # Education Degrees
    EDUCATION = ['BE','B.E.', 'B.E', 'BS','B.S','B.Com','BCA','ME','M.E', 'M.E.', 'M.S','B.com','10','10+2','BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'SSC', 'HSC', 'C.B.S.E','CBSE','ICSE', 'X', 'XII','10th','12th',' 10th',' 12th','Bachelor of Arts in Mathematics','Master of Science in Analytics','Bachelor of Business Administration','Major: Business Management']
       
    def extract_education(text):
        nlp_text = nlp(text)

        # Sentence Tokenizer
        nlp_text = [sent.text.strip() for sent in nlp_text.sents]


        edu = {}
        # Extract education degree
        for index, t in enumerate(nlp_text):
            for tex in t.split():
                # Replace all special symbols
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex in EDUCATION and tex not in STOPWORDS:
                    edu[tex] = t + nlp_text[index + 1]

        # Extract year
        education = []
        for key in edu.keys():
            year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
            if year:
                education.append((key, ''.join(year[0])))
            else:
                education.append(key)
        return education

    def extract_skills(resume_text):
        nlp_text = nlp(resume_text)
        noun_chunks = nlp_text.noun_chunks

        # removing stop words and implementing word tokenization
        tokens = [token.text for token in nlp_text if not token.is_stop]
        
        # reading the csv file
        data = pd.read_csv("https://github.com/Siddhant1803/Resume_-Classification_-_Parser-/blob/main/Resume.csv") 
        
        # extract values
        skills = list(data.columns.values)
        
        skillset = []
        
        # check for one-grams (example: python)
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
        
        # check for bi-grams and tri-grams (example: machine learning)
        for token in noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
        
        return [i.capitalize() for i in set([i.lower() for i in skillset])]
    def string_found(string1, string2):
        if re.search(r"\b" + re.escape(string1) + r"\b", string2):
            return True
        return False

    def extract_entity_sections_grad(text):
        '''
        Helper function to extract all the raw text from sections of resume specifically for 
        graduates and undergraduates
        :param text: Raw text of resume
        :return: dictionary of entities
        '''
        text_split = [i.strip() for i in text.split('\n')]
        entities = {}
        key = False
        for phrase in text_split:
            if len(phrase) == 1:
                p_key = phrase
            else:
                p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_GRAD)
            try:
                p_key = list(p_key)[0]
            except IndexError:
                pass
            if p_key in cs.RESUME_SECTIONS_GRAD:
                entities[p_key] = []
                key = p_key
            elif key and phrase.strip():
                entities[key].append(phrase)
        return entities
    
    # Function to extract experience details
    def expDetails(Text):
        global sent
       
        Text = Text.split()
       
        for i in range(len(Text)-2):
            Text[i].lower()
            
            if Text[i] ==  'years':
                sent =  Text[i-2] + ' ' + Text[i-1] +' ' + Text[i] +' '+ Text[i+1] +' ' + Text[i+2]
                l = re.findall(r'\d*\.?\d+',sent)
                for i in l:
                    a = float(i)
                return(a)
                return (sent)

    def extract_experience(resume_text):
        '''
        Helper function to extract experience from resume text
        :param resume_text: Plain resume text
        :return: list of experience
        '''
        wordnet_lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # word tokenization 
        word_tokens = nltk.word_tokenize(resume_text)

        # remove stop words and lemmatize  
        filtered_sentence = [w for w in word_tokens if not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words] 
        sent = nltk.pos_tag(filtered_sentence)

        # parse regex
        cp = nltk.RegexpParser('P: {<NNP>+}')
        cs = cp.parse(sent)
        
        # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
        #     print(i)
        
        test = []
        
        for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
            test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))

        # Search the word 'experience' in the chunk and then print out the text after it
        x = [x[x.lower().index('experience') + 10:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
        return x


    def extract_dob(text):
            
        result1=re.findall(r"[\d]{1,2}/[\d]{1,2}/[\d]{4}",text)
        result2=re.findall(r"[\d]{1,2}-[\d]{1,2}-[\d]{4}",text)           
        result3= re.findall(r"[\d]{1,2} [ADFJMNOSadfjmnos]\w* [\d]{4}",text)
        result4=re.findall(r"([\d]{1,2})\.([\d]{1,2})\.([\d]{4})",text)
                    
        l=[result1,result2,result3,result4]
        for i in l:
            if i==[]:
                continue
            else:
                return i

    def extract_text_from_docx(path):
        '''
        Helper function to extract plain text from .docx files
        :param doc_path: path to .docx file to be extracted
        :return: string of extracted text
        '''
        if path.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            temp = docx2txt.process(path)
            return temp

    def display(docx_path):
        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')

    df = pd.DataFrame(columns=['Name','Mobile No.', 'Email','DOB','Education Qualifications','Skills','Experience (Years)'], dtype=object)
    
    st.title("RESUME PARSER")
        
    st.subheader('Upload Resume (Single File Accepted) 👇')
    upload_file3 = st.file_uploader('', type= ['docx'], accept_multiple_files=False)

    st.write('*Note: For different Resumes Results Reupload')    
    
    if upload_file3 is not None:
        displayed=display(upload_file3)
        
        i=0
        text = extract_text_from_docx(upload_file3)
        tokText = tokenText(text)
        df.loc[i,'Name']=extract_name(tokText)
        df.loc[i,'Mobile No.']=extract_mobile_number(text)
        df.loc[i,'Email']=extract_email(text)
        df.loc[i,'DOB']=extract_dob(text)
        df.loc[i,'Education Qualifications']=extract_education(text)
        df.loc[i,'Skills']=extract_skills(text)
        df.loc[i,'Experience (Years)']=expDetails(text) 
        experience_list1=extract_entity_sections_grad(text) 

        if 'experience' in experience_list1:
            experience_list=experience_list1['experience']
            df.loc[i,'Last Position']=extract_experience(text)
        else:
            df.loc[i,'Last Position']='NA'
           

        st.header("**Resume Analysis**")
        st.success("Hello "+ df['Name'][0])

        col1, col2 = st.columns(2)
 
        with col1:

            st.header("Basic info")
            try:        
                st.subheader('Name: '+ df['Name'][0])
                st.subheader('Experience (Years): ' + str(df['Experience (Years)'][0]))
                st.subheader('Last Position: ' + str(df['Last Position'][0]))
                st.subheader('Education: ' + str(df['Education Qualifications'][0]))
                st.subheader('Email: ' + str(df['Email'][0]))
                st.subheader('Contact: ' + str(df['Mobile No.'][0]))
                st.subheader('Date of Birth: ' + str(df['DOB'][0]))
            except:
                pass

            expander = st.expander("See Resume")
            expander.write(displayed)    

        with col2:
            st.header("**Skills Analysis💡**")
            ## shows skill
            keywords = st_tags(label='### Skills that'+ df['Name'][0] + ' have',
            text=' -- Skills',value=df['Skills'][0],key = '1')
