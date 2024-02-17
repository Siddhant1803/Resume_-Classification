import pandas as pd
import streamlit as st
import hydralit_components as hc
import docx2txt
import pdfplumber
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
stop=set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import pickle
# load pre-trained model
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import Matcher
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

mfile = BytesIO(requests.get('https://github.com/Siddhant1803/Resume_Classification/edit/main/resume.py?raw=true').content)
model = load(mfile)

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')

#specify the primary menu-defination
menu_data = [
    {'icon':"fa fa-address-card",'label':'Resume Classification'},
    {'icon':"far fa-file-word",'label':'Resume Parser'},
]
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_defination = menu_data,
    override_theme = over_theme,
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
    tab1,tab2 = st.tabs(["📝 About Project","📉 About Data"])
    with tab1:
         st.title('About Project')

            st.subheader('Resume Classificaiton')

            st.header(f"Business Objective: The document classification solution should significantly reduce the manual human effort in the HRM. It should achieve a higher level of accuracy and automation with minimal human intervention.")

            st.image("https://www.bing.com/images/search?view=detailV2&ccid=4kWTbAjZ&id=F18A23CD4214BEFA96E7364DE7D3A5B0DCC0348E&thid=OIP.4kWTbAjZPun5sAwXIuA54QHaD4&mediaurl=https%3A%2F%2Fres.cloudinary.com%2Frootworks%2Fimage%2Fupload%2Fc_fill%2Cf_auto%2Cg_face%3Aauto%2Ch_630%2Cw_1200%2Fv1%2Fin_the_loop%2F2022-07-08%2Fremote-hiring-tips-itl-jul-aug-22_ttibpn&exph=630&expw=1200&q=resumes+coming+out+of+laptops+image&simid=607991993388856901&form=IRPRST&ck=FA57ECA78F27524282E50E0713FB133B&selectedindex=17&itb=0&ajaxhist=0&ajaxserp=0&pivotparams=insightsToken%3Dccid_LTS6Fz58*cp_0AE28F64A0A80744F0665AB76838060E*mid_795293533BA412F27A4608294E35326E7FBF3F63*simid_608054257537402558*thid_OIP.LTS6Fz58XBMS9dJNiXt9agHaGS&vt=0&sim=11&iss=VSI&ajaxhist=0&ajaxserp=0")

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
        st.title("About Data")
        st.subheader("Exploring through data")

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


# In[6]:


def mostcommon_words(cleaned,i):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(cleaned)
    mostcommon=FreqDist(cleaned.split()).most_common(i)
    return 


# In[7]:


def display_wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    a=px.imshow(wordcloud)
    st.plotly_chart(a)


# In[8]:


def display_words(mostcommon_small):
    x,y=zip(*mostcommon_small)
    chart=pd.DataFrame({'keys': x,'values': y})
    fig=px.bar(chart,x=chart['keys'],y=chart['values'],height=700,width=700)
    st.plotly_chart(fig)


# In[9]:


def main():
    st.title('DOCUMENT CLASSIFICATION')
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed=display(doc_file)

                cleaned=preprocess(display(doc_file))
                predicted= model.predict(vectors.transform([cleaned]))

                string='The Uploaded Resume is belongs to '+predicted[0]
                st.header(string)

                st.subheader('WORDCLOUD')
                display_wordcloud(mostcommon_words(cleaned,100))

                st.header('Frequency of 20 Most Common Words')
                display_words(mostcommon_words(cleaned,20))

if __name__ == '__main__':
    main()


# In[ ]:




