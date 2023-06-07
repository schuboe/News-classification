# News-classification
Code and datasets for text classification of news articles (language detection, topic identification, web scraping, Streamlit app)

This project was carried out as part of the Data Science course at WBS Coding School (Feb-June 2023).

**Text classification** refers to the categorization of text documents based on their content. In this project, I addressed two text classification problems: language detection and topic identification.

For **language detection**, I trained a multinomial Naive Bayes classifier (using unigrams and bigrams of characters within words as features). The model can detect English, German, French, and Spanish. The folder "language_detection" contains the notebook (language_detection_NB.ipynb) and the dataset used for training (Language_Detection.csv).

**Topic identification** models were trained for German and English news articles, distinguishing the following topic categories: Business, Entertainment & Arts, Media, Panorama, Politics, Science, Sport, and Tech. For German, two models were trained based on a dataset of approximately 10,000 news articles: a Support Vector Machine (using TF-IDF features) and a Deep Learning model (using word embeddings). For English, a Support Vector Machine was trained (using TF-IDF features) on a dataset of approximately 1,500 news articles. The folder "topic_identification" contains the respective notebooks (news_de_SVM.ipynb, news_de_DL.ipynb, and news_en_SVM.ipynb) and datasets for training (articles.csv and en_news_data.csv).

Based on the text classification models, I created a **Streamlit app** that detects the language of an input text and 

Web scraping
