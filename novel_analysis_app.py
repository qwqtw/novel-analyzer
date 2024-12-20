import jieba
import string
import streamlit as st
from functions import (
    load_stop_words,
    basic_statistics,
    load_text,
    preprocess_text,
    word_frequency_analysis,
    generate_wordcloud,
    analyze_keyword_distribution,
)


def detect_language(text):
    # A simple heuristic: if the text contains Chinese characters, it's Chinese
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "chinese"
    return "english"


def preprocess_text_based_on_language(text, stop_words):
    language = detect_language(text)

    if language == "chinese":
        # Use jieba for Chinese tokenization
        words = jieba.lcut(text)
    else:
        # For English, split by whitespace and remove punctuation
        words = (
            text.lower().translate(str.maketrans("", "", string.punctuation)).split()
        )

    # Remove stop words and non-alphabetic tokens (for English)
    filtered_words = [
        word for word in words if word.isalpha() and word not in stop_words
    ]
    return filtered_words


def main():
    st.title("Novel Text Analysis")
    st.write(
        "Upload a .txt file for analysis of word frequency, word cloud, and keyword distribution."
    )

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        # Read the content of the uploaded file as a string
        text = uploaded_file.read().decode(
            "utf-8"
        )  # Decode the file content to a string

        # Display basic statistics
        stats = basic_statistics(text)
        st.subheader("Basic Statistics")
        st.text(stats)  # Display the basic statistics on the UI

        stop_words = load_stop_words()

        # Preprocess the text based on detected language (Chinese or English)
        words = preprocess_text_based_on_language(text, stop_words)

        # Word Frequency Analysis
        st.header("Word Frequency Analysis")
        word_freq_image = word_frequency_analysis(words)
        st.image(word_freq_image, caption="Word Frequency Analysis")

        # Word Cloud Generation
        st.header("Word Cloud")
        wordcloud_image = generate_wordcloud(words)
        st.image(wordcloud_image, caption="Word Cloud")

        # Keyword Distribution
        st.header("Keyword Frequency Distribution")
        keyword_dist_image = analyze_keyword_distribution(words, [])
        st.image(keyword_dist_image, caption="Keyword Frequency Distribution")


if __name__ == "__main__":
    main()
