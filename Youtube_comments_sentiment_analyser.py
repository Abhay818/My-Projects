import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import regex as re
from textblob import TextBlob
from urllib.parse import urlparse, parse_qs
from collections import Counter
import altair as alt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import requests 
import plotly.graph_objects as go  #  Plotly graph_objects module for pie chart
# Downloading the NLTK stopwords data 
# nltk.download('stopwords')



# Define stopwords set
stop_words = set(stopwords.words('english'))



# API key
API_KEY = 'AIzaSyCA1AtHQHt9Be_5AiUH_XFVP6gwhQjceQs'  



# Converting URL into video_ID
def get_video_id(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v')

    if video_id:
        return video_id[0]
    else:
        raise ValueError("Invalid YouTube URL: No video ID found")



# Building the API client
def build_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)

#Function for Fetching video comments
def fetch_video_comments(video_id):
    youtube = build_youtube_client()
    comments = []
    nextPageToken = None
    comment_counter = 0

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            comment_counter += 1

            if comment_counter >= 100:
                break

        nextPageToken = response.get("nextPageToken")

        if not nextPageToken or comment_counter >= 100:
            break

    return comments




#This is for Fetching all the video details using API call
def fetch_video_details(video_id):
    youtube = build_youtube_client()
    video = youtube.videos().list(
        part="snippet, statistics",
        id=video_id
    ).execute()

    video_details = {}
    if "items" in video and len(video["items"]) > 0:
        video = video["items"][0]
        video_details["Title"] = video["snippet"]["title"]
        video_details["Description"] = video["snippet"]["description"]
        video_details["Likes"] = int(video["statistics"].get("likeCount", 0))
        video_details["Dislikes"] = int(video["statistics"].get("dislikeCount", 0))
        video_details["Views"] = int(video["statistics"].get("viewCount", 0))
        video_details["Total Comments"] = int(video["statistics"].get("commentCount", 0))
        video_details["Date Published"] = video["snippet"]["publishedAt"]
        video_details["Thumbnail URL"] = video["snippet"]["thumbnails"]["high"]["url"]  # Use "high" quality thumbnail"
    return video_details





# Sentiment Analysis using TextBlob
def analyze_sentiment_textblob(comment):
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"





# most negative words
def find_most_negative_words(df, limit=11):
    all_comments_text = " ".join(df["Comment"].tolist())
    blob = TextBlob(all_comments_text)
    word_polarity = {word: TextBlob(word).sentiment.polarity for word in blob.words}

    # Sorting (most negative first)
    most_negative_words = sorted(word_polarity.items(), key=lambda x: x[1])[:limit]

    return most_negative_words

# most positive words
def find_most_positive_words(df, limit=11):
    all_comments_text = " ".join(df["Comment"].tolist())
    blob = TextBlob(all_comments_text)
    word_polarity = {word: TextBlob(word).sentiment.polarity for word in blob.words}

    # Sorting (most positive first)
    most_positive_words = sorted(word_polarity.items(), key=lambda x: x[1], reverse=True)[:limit]

    return most_positive_words




# most common words
def find_most_common_words(df, limit=20):
    all_comments_text = " ".join(df["Comment"].tolist())
    words = re.findall(r'\b\w+\b', all_comments_text.lower())  # Tokenize words in the comments
    words = [word for word in words if word not in stop_words]  # Filter out stopwords
    word_counter = Counter(words)
    most_common_words = word_counter.most_common(limit)
    return most_common_words





# Extract emojis from text and putting them in a list
def extract_emojis(text):
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]")
    return [c for c in text if re.match(emoji_pattern, c)]

# Counting emojis in comments
def count_emojis(comments):
    emoji_list = []
    for comment in comments:
        emojis_in_comment = extract_emojis(comment)
        emoji_list.extend(emojis_in_comment)
    return emoji_list

# Now emoji table created from emoji list
def create_emoji_table(emoji_list):
    emoji_counter = Counter(emoji_list)
    emoji_df = pd.DataFrame(emoji_counter.items(), columns=['Emoji', 'Count'])
    emoji_df = emoji_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    return emoji_df

# Here emoji bar plot is created using Altair
def create_emoji_bar_plot(emoji_df):
    chart = alt.Chart(emoji_df).mark_bar().encode(
        x=alt.X('Emoji', axis=alt.Axis(labelAngle=0)),  # Set labelAngle to 0 for horizontal orientation
        y='Count'
    ).properties(
        width=alt.Step(30)
    )
    return chart





# word cloud (basically group of words ) from most common words
def create_word_cloud(df_common_words):
    # Combine the most common words into a single string
    common_words_text = " ".join(df_common_words["Word"].tolist())

    wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(common_words_text)

    # Here matplotlib is used for displaying the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()




# Converting comments to DataFrame and perform sentiment analysis (all the tables as dataframes are created here)
def convert_to_dataframe(video_comments):
    df = pd.DataFrame(video_comments, columns=["Comment"])
    df["Polarity"] = df["Comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

    #most negative and most positive words
    most_negative_words = find_most_negative_words(df)
    most_positive_words = find_most_positive_words(df)

    # most negative and most positive words with frequency and polarity
    df_most_negative = pd.DataFrame(most_negative_words, columns=["Word", "Polarity"])
    df_most_positive = pd.DataFrame(most_positive_words, columns=["Word", "Polarity"])

    # Counting the frequency of words and adding to the dataframes
    df_most_negative["Frequency"] = df_most_negative["Word"].apply(lambda x: df["Comment"].str.count(x).sum())
    df_most_positive["Frequency"] = df_most_positive["Word"].apply(lambda x: df["Comment"].str.count(x).sum())

    return df, df_most_negative, df_most_positive





# best comments (highest positive polarity)
def find_best_comments(df, limit=5):
    df_best_comments = df[df['Sentiment'] == 'Positive'].sort_values(by='Polarity', ascending=False).head(limit)
    return df_best_comments

# worst comments (lowest negative polarity)
def find_worst_comments(df, limit=5):
    df_worst_comments = df[df['Sentiment'] == 'Negative'].sort_values(by='Polarity').head(limit)
    return df_worst_comments




# pie chart 
def create_polarity_pie_chart(df):
    polarity_counts = df['Sentiment'].value_counts()
    labels = polarity_counts.index
    values = polarity_counts.values

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(
        title="Comment Polarity Distribution",
        title_x=0.33, 
        title_y=0.95,  
        showlegend=True

    )

    return fig





# Main function
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="YouTube Comment Sentiment Analyzer")
    st.title("YouTube Comment Sentiment Analyzer")


    st.write("Welcome to the YouTube Comment Sentiment Analyzer! This tool allows you to analyze the sentiment of comments on any YouTube video. Simply enter the URL of the YouTube video you want to analyze, and click the 'Analyze' button. The tool will fetch the comments, perform sentiment analysis, and display the results. Here's what you can expect from the YouTube Comment Sentiment Analyzer:\n"
             "\n"
             "1. **Video Details**\n"             
             "2. **Sentiment Analysis Results**\n"            
             "3.  **Average polarity of comments-- Positive, Negative and Overall**\n"             
             "4. **Pie chart for Polarity Distribution of comments**\n"             
             "5. **Most Positive and Negative Comments**\n"            
             "6. **Most Positive , Negative and Common Words**: \n"            
             "7. **Word Cloud for most common words used**\n"             
             "8. **Emoji Count-  types of emoji used and their count**\n"
             "9. **Bar plot for emoji frequency.**\n"
             "\n"
             "Explore the sentiment of YouTube video comments with this powerful tool! ")
   
   
    st.write("")
    youtube_url = st.text_input("Enter YouTube URL:")

    if st.button("Analyze"):
        try:
            video_id = get_video_id(youtube_url)
            video_comments = fetch_video_comments(video_id)
            df, df_most_negative, df_most_positive = convert_to_dataframe(video_comments)
            video_details = fetch_video_details(video_id)
            emoji_list = count_emojis(video_comments)
            emoji_df = create_emoji_table(emoji_list)

            st.write("")
            #st.markdown("<h2 style='text-align: center;'>Dashboard</h2>", unsafe_allow_html=True)
            #st.write("")
            #st.write("")
            
            st.markdown("<h3 style='text-align: center;'>Video Details:</h3>", unsafe_allow_html=True)
            
            st.dataframe(pd.DataFrame.from_dict(video_details, orient='index', columns=['Value']), use_container_width=True)
            thumbnail_url = video_details["Thumbnail URL"]
            response = requests.get(thumbnail_url, stream=True)
            thumbnail_image = Image.open(response.raw)
            st.image(thumbnail_image, use_column_width=True)
            

            st.write("")
            st.write("")

            
            
            #st.markdown("<h3 style='text-align: center;'>Video Thumbnail</h3>", unsafe_allow_html=True)
            
            st.write("")
            st.write("")

            st.markdown("<h3 style='text-align: center;'>Sentiment Analysis Results</h3>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            st.write("")
            st.write("")



            # Calculate average polarity for positive and negative comments
            avg_polarity_positive = df[df['Sentiment'] == 'Positive']['Polarity'].mean()
            avg_polarity_negative = df[df['Sentiment'] == 'Negative']['Polarity'].mean()
            # Calculate overall average polarity
            overall_avg_polarity = df['Polarity'].mean()

            # Display average polarity for positive and negative comments
            st.markdown("<h4 style='text-align: center;'>Average Polarity</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Average Polarity (Positive Comments): {:.2f}</p>".format(avg_polarity_positive), unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Average Polarity (Negative Comments): {:.2f}</p>".format(avg_polarity_negative), unsafe_allow_html=True)

            # Display overall average polarity
            st.markdown("<h4 style='text-align: center;'>Overall Average Polarity</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Overall Average Polarity: {:.2f}</p>".format(overall_avg_polarity), unsafe_allow_html=True)



            st.write("")
            st.write("")
            st.write("")
            st.write("")


            # pie chart 
            st.markdown("<h4 style='text-align: center;'>Polarity Distribution</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_polarity_pie_chart(df), use_container_width=True)
        

            st.write("")
            st.write("")


            # Find best and worst comments
            df_best_comments = find_best_comments(df)
            df_worst_comments = find_worst_comments(df)
            # Display the best and worst comments in tables
            st.markdown("<h4 style='text-align: center;'>Best Positive Comments</h4>", unsafe_allow_html=True)
            st.dataframe(df_best_comments, use_container_width=True)
            st.write("")
            st.markdown("<h4 style='text-align: center;'>Worst Negative Comments</h4>", unsafe_allow_html=True)
            st.dataframe(df_worst_comments, use_container_width=True)


            st.write("")
            st.write("")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h4>Most Negative Words: </h4>", unsafe_allow_html=True)
                st.dataframe(df_most_negative)

            with col2:
                st.markdown("<h4>Most Positive Words: </h4>", unsafe_allow_html=True)
                st.dataframe(df_most_positive)

            st.write("")
            st.write("")

            # most common words in the comments
            most_common_words = find_most_common_words(df, limit=20)
            most_common_words_df = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
            st.markdown("<h4>Most Common Words </h4>", unsafe_allow_html=True)
            st.dataframe(most_common_words_df)

            st.write("")
            st.write("")

            # word cloud 
            st.markdown("<h4 style='text-align: center;'>Word Count for most words used </h4>", unsafe_allow_html=True)
            create_word_cloud(most_common_words_df)


            st.write("")
            st.write("")


            #emoji table
            st.markdown("<h4>Emoji's Count </h4>", unsafe_allow_html=True)
            st.write(emoji_df)


            st.write("")
            st.write("")

            # vertical bar plot for emoji frequency
            st.markdown("<h4 style='text-align: center;'>Emoji Frequency Bar Plot</h4>", unsafe_allow_html=True)
            st.altair_chart(create_emoji_bar_plot(emoji_df), use_container_width=True)



            # Footer Section
            st.markdown("---")

            st.markdown("<h3 style='text-align: center;'>Thank You for Using YouTube Comment Sentiment Analyzer!</h3>", unsafe_allow_html=True)
            st.write("")
            st.write("")
            st.write("Made with ❤️ by Your Abhay Singh Bisht, Bhoomi Taneja, Aryan")
            st.write("Follow me on [Twitter](https://twitter.com/AbhaySinghBish7)")

        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
