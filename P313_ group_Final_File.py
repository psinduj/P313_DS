#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from nltk.corpus import stopwords
import nltk
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud,STOPWORDS
import string
import re


# In[4]:


#Load dataset
data=pd.read_excel('hotel_reviews.xlsx')
data.head()


# In[5]:


data.drop(columns={'@'},inplace=True)
data.head(3)


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data[data.duplicated()]


# In[9]:


data.info()


# In[10]:


data['Rating'].value_counts()


# In[11]:


# Plotting the distribution of ratings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
data['Rating'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[12]:


data.head()


# In[13]:


#Creating a new column 'Length' that will contain the length of the string 'Review' column
data['Length']=data['Review'].apply(len)


# In[14]:


data.head()


# In[15]:


data['Length'].describe()


# In[16]:


import plotly.express as px
px.histogram(data,data['Length'],color='Rating',color_discrete_sequence=['lightpink', 'lightgrey', 'lightgreen', 'lightskyblue', 'lightyellow'],title="Review Length Distributions")


# In[17]:


g = sns.FacetGrid(data=data, col='Rating')
g.map(plt.hist, 'Length', color='#973aa8')


# In[18]:


#Finding the percentage distribution of each rating- we will divide the number of records for each rating by total number of record
print(f"Rating value count - percentage distribution: \n{round(data['Rating'].value_counts()/data.shape[0]*100,2)}")


# In[19]:


rating_counts = data['Rating'].value_counts()


# In[20]:


# Plotting a pie chart for ratings
plt.figure(figsize=(6, 6))
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['lightpink', 'lightgrey', 'lightgreen', 'lightskyblue', 'lightyellow'])

plt.title('Distribution of Ratings')
plt.legend(loc= 'upper right')
plt.show()


# In[21]:


booking_sites = data['Review'].str.extract(r'(\w+\.com)')[0].value_counts()
booking_sites


# In[22]:


plt.figure(figsize=(15,12))
sns.barplot(y=booking_sites.head(10).index, x=booking_sites.head(10))
plt.title('top 10 websites for booking an hotel')
plt.xlabel(' No of customers using the website to book the hotel')
plt.ylabel('websites name')
plt.show()


# In[ ]:


#Expensive=data['Review'].str.extract(r'(expensive)')[0].value_counts()
#Affordable=data['Review'].str.extract(r'(affordable)')[0].value_counts()
#cost_ratings=([Expensive,Affordable])
#cost_ratings
#plt.figure(figsize=(4,4))
#sns.barplot(cost_ratings)
#plt.title('cost reduction')
#plt.xlabel(' classification based on cost')
#plt.ylabel('no of word counts')
#plt.show()


# In[28]:


word1 = 'expensive'
word2 = 'cheap'
# Extract occurrences of the two words in each review
data['Word1_Count'] = data['Review'].apply(lambda x: x.lower().count(word1))
data['Word2_Count'] = data['Review'].apply(lambda x: x.lower().count(word2))


# In[29]:


# Visualize using a bar plot
plt.figure(figsize=(10, 6))

sns.barplot(x=data['Rating'], y=data['Word1_Count'], color='blue', label=word1)

# Bar plot for Word2_Count
sns.barplot(x=data['Rating'], y=data['Word2_Count'], color='lightgreen', label=word2)

plt.title(f'Cost of an hotel wrt to Rating')
plt.xlabel('Rating')
plt.ylabel('value for occurence of Word Count')
plt.legend()
plt.show()


# In[30]:


Expensive


# In[31]:


data


# In[32]:


# Creating a Word Cloud of reviews
all_reviews = ' '.join(data['Review'])
wordcloud = WordCloud(width=800, height=400, background_color='Black').generate(all_reviews)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()


# In[33]:


import nltk
nltk.download('vader_lexicon')


# In[34]:


# Analyze sentiment of reviews
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
data['sentiment_score'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])


# In[35]:


data.head(5)


# In[36]:


# Plotting the distribution of sentiment scores
plt.subplot(1, 2, 2)
data['sentiment_score'].plot(kind='hist', bins=20, color='lightgreen')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[37]:


#  Sentiment Analysis
data['Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral')
data['Sentiment']


# In[38]:


s_counts2 = data['Sentiment'].value_counts()
s_counts2


# In[39]:


#  Sentiment Analysis (Pie chart)
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 4)
data['Sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['Pink', 'green', 'blue'])
plt.title('Sentiment Analysis')
plt.tight_layout()
plt.show()


# In[40]:


s_counts2 = data['Sentiment'].value_counts()
s_counts2


# In[41]:


def no_of_words(text):
  words=text.split()
  word_count=len(words)
  return word_count


# In[42]:


data['word_count']=data['Review'].apply(no_of_words)


# In[43]:


pos_reviews=data[data.Sentiment=='Positive']
pos_reviews.head()


# In[44]:


neg_reviews=data[data.Sentiment=='Negative']
neg_reviews.head()


# In[45]:


#Positive wordCloud
text= ' '.join([word for word in pos_reviews['Review']])
plt.figure(figsize=(20,15),facecolor=None)
pos_wordcloud=WordCloud(max_words=500,width=1600,height=800).generate(text)


#Negative wordCloud
text= ' '.join([word for word in neg_reviews['Review']])
plt.figure(figsize=(20,15),facecolor=None)
neg_wordcloud=WordCloud(max_words=500,width=1600,height=800).generate(text)

# Visualizations
plt.figure(figsize=(12, 6))

# Positive Word Cloud
plt.subplot(1, 2, 1)
plt.imshow(pos_wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')

# Negative Word Cloud
plt.subplot(1, 2, 2)
plt.imshow(neg_wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')

plt.tight_layout()
plt.show()


# In[46]:


pos_reviews.Sentiment.replace("Positive",1,inplace=True)


# In[47]:


from collections import Counter
count=Counter()
for text in pos_reviews['Review'].values:
  for word in text.split():
    count[word] +=1
count.most_common(15)


# In[48]:


pos_words=pd.DataFrame(count.most_common(15))
pos_words.columns=['word','count']
pos_words.head()


# In[49]:


import plotly.express as px
px.bar(pos_words,x='count',y='word',title='Common words in positive reviews',color='word')


# In[50]:


neg_reviews.Sentiment.replace("Negative",2,inplace=True)


# In[51]:


from collections import Counter
count=Counter()
for text in neg_reviews['Review'].values:
  for word in text.split():
    count[word] +=1
count.most_common(15)


# In[52]:


neg_words=pd.DataFrame(count.most_common(15))
neg_words.columns=['word','count']
neg_words.head()


# In[53]:


px.bar(neg_words,x='count',y='word',title='Common words in negative reviews',color='word')


# # Text Cleaning and Preprocessing

# In[54]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))


# In[55]:


def data_processing(text):
  # Removing emojis
  def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
  text=text.lower()                                                     #Converting the entire text in the reviews to lowercase
  text=re.sub(r"http\S+|www\S+|https\S+", '',text, flags= re.MULTILINE) #Removing url's
  text=re.sub(r'\S+@\S','',text)                                        #Removing the emails from reviews
  text=re.sub(r'\d+','',text)                                           #Removing the digits from reviews
  text=text.strip()                                                     #removing extra space from the reviews
  text=remove_emoji(text)                                               #Removing emojis
  text=re.sub(r'[^\w\s]','',text)                                       #remove all the punctuations from the reviews
  text_tokens=word_tokenize(text)
  filtered_text=[w for w in text_tokens if not w in stop_words]
  return " ".join(filtered_text)


# In[56]:


data['Clean_Review']=data['Review'].apply(data_processing)


# In[57]:


data.head()


# In[58]:


import nltk
nltk.download('wordnet')
nltk.download('stopwords')


# In[59]:


def cleaning(text):
    clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]

#lemmatize the word
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)


# In[ ]:


data['Clean_Review']= data['Clean_Review'].apply(cleaning)


# In[ ]:


data


# In[ ]:


data['length'] = data['Clean_Review'].apply(len)


# In[ ]:


original_length=data['Length'].sum()
new_length = data['length'].sum()

print('Total text length before cleaning: {}'.format(original_length))
print('Total text length after cleaning: {}'.format(new_length))


# # Spliting the data into training and testing

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


X=data['Clean_Review']
Y=data['Sentiment']


# # Word Embedding:
#   + Tfidf Vectorizer
#   + Count Vectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer()
X=vect.fit_transform(data['Clean_Review'])


# In[ ]:


X


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[ ]:


print("Size of x_train",(x_train.shape))
print("Size of x_test",(x_test.shape))
print("Size of y_train",(y_train.shape))
print("Size of y_test",(y_test.shape))


# In[ ]:


labelEncoder=LabelEncoder()
y_train=labelEncoder.fit_transform(y_train)
y_test=labelEncoder.transform(y_test)

labels=labelEncoder.classes_.tolist()
print(labels)


# In[ ]:


y_train


# In[ ]:


# Plotting the distribution of ratings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
data['Sentiment'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title('Distribution of Sentiment')
plt.xlabel('Rating')
plt.ylabel('Count')


# In[ ]:


np.bincount(y_train)


# # Logistic Regression Model

# In[ ]:


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
logreg_pred=logreg.predict(x_test)
logreg_acc=accuracy_score(logreg_pred,y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[ ]:


print(confusion_matrix(y_test,logreg_pred))
print("\n")
print(classification_report(y_test,logreg_pred))


# # SMOTE Over Sampling

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler()
ros_X_train,ros_y_train=ros.fit_resample(x_train,y_train)


# In[ ]:


from collections import Counter
print("Before sampling class distribution:",Counter(y_train))
print("After sampling class distribution:",Counter(ros_y_train))


# In[ ]:


clf1=LogisticRegression(random_state=0)
clf1.fit(ros_X_train,ros_y_train)
clf1_pred1=clf1.predict(x_test)
clf1_acc=accuracy_score(clf1_pred1,y_test)
print("Test accuracy: {:.2f}%".format(clf1_acc*100))


# In[ ]:


y_test.shape


# In[ ]:


print(confusion_matrix(y_test,clf1_pred1))
print("\n")
print(classification_report(y_test,clf1_pred1))


# # MultinomialNB Model

# In[ ]:


mnb=MultinomialNB()
mnb.fit(ros_X_train,ros_y_train)
mnb_pred=mnb.predict(x_test)
mnb_acc=accuracy_score(mnb_pred,y_test)
print("Test accuracy: {:.2f}%".format(mnb_acc*100))


# In[ ]:


from sklearn.metrics import classification_report
print(confusion_matrix(y_test,mnb_pred))
print("\n")
print(classification_report(y_test,mnb_pred))


# # SVC Model

# In[ ]:


svc=LinearSVC()
svc.fit(ros_X_train,ros_y_train)
svc_pred=svc.predict(x_test)
svc_acc=accuracy_score(svc_pred,y_test)
print("Test accuracy: {:.2f}%".format(svc_acc*100))


# In[ ]:


from sklearn.metrics import classification_report
print(confusion_matrix(y_test,svc_pred))
print("\n")
print(classification_report(y_test,svc_pred))


# # RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
rf.fit(ros_X_train,ros_y_train)


# In[ ]:


Y_predi=rf.predict(x_test)


# In[ ]:


rf_acc=accuracy_score(y_test,Y_predi)


# In[ ]:


rf_acc=accuracy_score(y_test,Y_predi)
rf_acc


# Negative-0
# Neutral-1
# Positive-2

# In[ ]:


rev = ["nice place with good food and accomodation"]
rev_v =vect.transform(rev)
logreg.predict(rev_v)


# In[ ]:


rev = ["this chips are ok for health"]
rev_v =vect.transform(rev)
logreg.predict(rev_v)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




