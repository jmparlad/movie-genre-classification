# Movie genre classification based on synopses with BERT
This repository contains the files used for the project. A summary of the files is presented below:
- IMDb_All_Genres_etf_clean1.csv is the original data set downloaded from Kaggle to get the movie labels and iterate for the web scraping.
- Movie_Data.csv is the data set used for training after the web scraping has been carried out.
- web_scraping_script is the script that builds the data set used in the project as described in the report.
- main.py contains the main programming of the classification models together with their evaluation.
- bert.py, bert_truncated.py and bert_with_pooling.py are the modified scrpits from the library BELT (BERT for Longer Texts, https://github.com/mim-solutions/bert_for_longer_texts). Here, it is specified with comments the parts that were modified.
