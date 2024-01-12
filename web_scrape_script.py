import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Load dataset from kaggle:
df = pd.read_csv('IMDb_All_Genres_etf_clean1.csv')

count = 0
Title = []
Synopsis = []
Main_Genre = []
for i in range(len(df)):
    # Get movie title from dataset and generate URL:
    title = df.iloc[i].Movie_Title
    processed_title = title.lower()
    the_flag = 0
    if ':' in processed_title:
        processed_title = processed_title.replace(':', '')

    url_title = '_'.join(processed_title.split())
    url = 'https://www.rottentomatoes.com/m/' + url_title

    # Establish connection and gather the html code:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    soup_text = str(soup)

    # Set the limits for the desired part of the text:
    soup_text = soup_text.replace('<p data-qa="movie-info-synopsis" slot="content">', '_START_')
    soup_text = soup_text.replace('<button class="button--link" slot="ctaClose">Show Less</button>', '_END_')

    # Process the text and gather only the desired data:
    flag = 0
    synops_list = []
    for word in soup_text.split():
            if flag:
                if word != '':
                    synops_list.append(word)

            if word == '_START_':
                flag = 1
            elif '_END_' in word:
                flag = 0
            synops = ' '.join(synops_list)
            synops = synops.replace('</p> _END_', '')
    if synops != '':
        if synops is not np.nan:
            count += 1
            Title.append(title)
            Synopsis.append(synops)
            Main_Genre.append(df.iloc[i].main_genre)

# Create the new panda dataframe
d = {'Title': Title,
     'Synopsis': Synopsis,
     'Main_Genre': Main_Genre,
    }
new_df = pd.DataFrame(data=d)

print('The total number of synopses successfully gathered is ' + str(count) + '.')
new_df.head()
# Safe to .csv for quick access:
new_df.to_csv('Movie_Data', sep=',', index=False, encoding='utf-8')