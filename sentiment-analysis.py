import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from wordcloud import WordCloud

# Set page config
st.set_page_config(page_title="IKEA Sentiment Dashboard", layout="wide")

st.title("🛒 IKEA Customer Reviews Sentiment Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your IKEA reviews CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Data")
    st.dataframe(df)

    # Sentiment analysis
    st.subheader("🧠 Sentiment Analysis")

    def get_polarity(text):
        return TextBlob(text).sentiment.polarity

    df['polarity'] = df['review_text'].apply(get_polarity)
    df['sentiment'] = df['polarity'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    st.write("✅ Sentiment column added!")
    st.dataframe(df[['product_name', 'rating', 'review_text', 'sentiment']])

    # Word clouds
    st.subheader("🌈 Word Clouds by Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👍 Positive Reviews Word Cloud")
        positive_text = " ".join(df[df['sentiment'] == 'Positive']['review_text'].dropna().astype(str))
        if positive_text:
            wordcloud_pos = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
            fig_pos, ax_pos = plt.subplots()
            ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
            ax_pos.axis("off")
            st.pyplot(fig_pos)
        else:
            st.info("No positive reviews to show.")

    with col2:
        st.markdown("#### 👎 Negative Reviews Word Cloud")
        negative_text = " ".join(df[df['sentiment'] == 'Negative']['review_text'].dropna().astype(str))
        if negative_text:
            wordcloud_neg = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(negative_text)
            fig_neg, ax_neg = plt.subplots()
            ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
            ax_neg.axis("off")
            st.pyplot(fig_neg)
        else:
            st.info("No negative reviews to show.")

    # Sentiment count plot
    st.subheader("📊 Sentiment Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='sentiment', hue='sentiment', palette='coolwarm', legend=False, ax=ax1)
    st.pyplot(fig1)

    # Rating vs sentiment
    st.subheader("⭐ Ratings vs Sentiment")

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='sentiment', y='rating', hue='sentiment', palette='pastel', dodge=False, legend=False, ax=ax2)
    st.pyplot(fig2)

    # Download enriched data
    st.subheader("⬇️ Download Result")
    st.download_button("Download CSV with Sentiments", df.to_csv(index=False), file_name="ikea_reviews_with_sentiment.csv")
