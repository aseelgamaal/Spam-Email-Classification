import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import numpy as np
import pandas as pd 
import PIL as image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.set_page_config(page_title='Page_1', page_icon='::star::')


def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def input_model(chosen_model):

  
    if chosen_model == 'logistic regression':
        chosen_model = 1
    elif chosen_model == 'MultinomialNB':
        chosen_model = 3
    elif chosen_model == 'GaussianNB':
        chosen_model = 2
    elif chosen_model == 'BernoulliNB':
        chosen_model = 4
    elif chosen_model == 'decision tree':                                       
        chosen_model = 8
    elif chosen_model == 'Support Vector Machine':
        chosen_model = 6
    elif chosen_model == 'random forest':
        chosen_model = 5
    else:
          chosen_model = 7
    

    
    return chosen_model



with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs", "About","Contact"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )




job_vec= joblib.load(open("TfidfVectorizer","rb"))

#tfidf = pickle.load(open('vectorizer.pkl','rb'))





loaded_model1 = joblib.load(open('my_model_1','rb'))
loaded_model2 = joblib.load(open('my_model_2','rb'))
loaded_model3 = joblib.load(open('my_model_3','rb'))
loaded_model4 = joblib.load(open('my_model_4','rb'))
loaded_model5 = joblib.load(open('my_model_5','rb'))
loaded_model6 = joblib.load(open('my_model_6','rb'))
loaded_model7 = joblib.load(open('my__model_7','rb'))
loaded_model8 = joblib.load(open("my_model_8", 'rb'))




if choose=='Home':
  
  st.title("Email Spam Classifier")




 #st.header('Placement')
  link_3 = "https://lottie.host/02522e87-4bf2-4098-8282-95b08e3f26da/e9SmHgtKW2.json"
  lottie_link3 = "https://lottie.host/37865deb-f4f3-4b04-83b8-baa7b9f70e77/ZwmymW836O.json"
  link_4="https://lottie.host/56fe043f-e552-43fe-86da-8cb11b96f18e/S45MELw5HV.json"
  animation5 = load_lottie(lottie_link3)
  animation6= load_lottie(link_3)
  animation7=load_lottie(link_4)
  st.write('---')


  with st.container():
    
    right_column, left_column = st.columns(2)
    
    with right_column:
        input_sms = st.text_area("Enter the message")
        age = st.number_input('AGE : ',  min_value=18 , max_value=90 , step=1)

        name = st.text_input('Name:')
        
        gender = st.radio('Gender : ', ['F', 'M'])
        
         
        chosen_model = st.selectbox('Model for prediction : ', ('logistic regression','GaussianNB','MultinomialNB','BernoulliNB','random forest','Support Vector Machine','KNN', 'decision tree'))
        
  with left_column:
        st_lottie(animation6, speed=1, height=300, key="animation66")
        st_lottie(animation5, speed=1, height=300, key="animation55")

        


  if st.button('PREDICT'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input1 = job_vec.transform([transformed_sms])




    # 3. predict
    which = input_model(chosen_model)

    if which == 1:
         result = loaded_model1.predict(vector_input1)[0]
    elif which == 2:
        result = loaded_model2.predict(vector_input1.toarray())[0]
    elif chosen_model == 3:
        result = loaded_model3.predict(vector_input1.toarray())[0]
    elif chosen_model == 4:
        result = loaded_model4.predict(vector_input1.toarray())[0]
    elif chosen_model == 5:                                       
        result = loaded_model5.predict(vector_input1)[0]
    elif chosen_model == 6:
        result = loaded_model6.predict(vector_input1)[0]
    elif chosen_model == 7:
        result = loaded_model7.predict(vector_input1)[0]
    else:
        result = loaded_model8.predict(vector_input1)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
        st.write()        
        st.warning("⚠️  This is a spam  message!")
        st_lottie(animation7, speed=2, height=400, key="animation77")
        
    else:
        st.header("Not Spam")
        st.balloons()

elif choose=='About':
    st.write('# About Page')
    st.write('---')
    st.write("🎯💡 We use a publicly available dataset of sensor readings from industrial equipment, comprising features such as temperature, pressure, and vibration. The dataset provides a realistic environment for testing predictive maintenance algorithms and assessing their effectiveness in real-world scenarios.Through our experiments, we uncover valuable insights into the predictive power of different machine learning algorithms for maintenance prediction. We analyze the strengths and weaknesses of each model and provide recommendations for deploying predictive maintenance systems in practical settings. In conclusion, our AI notebook serves as a comprehensive guide to predictive maintenance using machine learning. By following along with the code examples and experiments, readers can gain a deeper understanding of predictive maintenance techniques and their applications in various industries.📞📧")

elif choose=='Contact': 
    st.write('# Contact ')
    st.write('---')
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        st.write('## Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') 
        Email=st.text_input(label='Please Enter Email')
        Message=st.text_input(label='Please Enter Your Message') 
        submitted = st.form_submit_button('Submit')
        if submitted:
          st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


elif choose=='Graphs':       
    st.write('# Classifier Graphs')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.image("output1.png")
    st.image("output2.png")
    st.image("output3.png")
    st.image("output4.png")
    st.image("output5.png")
    st.image("output6.png")
    st.image("output7.png")
    st.image("output8.png")
    st.image("output9.png")
    st.image("output10.png")
    st.image("output11.png")
    st.image("output12.png")
    st.image("output13.png")
    st.image("output14.png")
    st.image("output15.png")
    st.image("output16.png")
    st.image("output17.png")
    st.image("output18.png")
    st.image("output19.png")
    


   

#streamlit run deploy.py

#"C:\Users\arwam\OneDrive\Documents\AI project"



