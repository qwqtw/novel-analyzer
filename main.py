import os
import tempfile
import streamlit as st
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import re
from functions import *  # Assuming your functions are in a file named 'functions.py'

# Set up Streamlit page
st.set_page_config(page_title="Novel Text Analyzer", layout="wide")


def display_word_frequency_chart(common_words):
    words, freqs = zip(*common_words)
    words_pos = {word: pseg.cut(word).__next__().flag for word, _ in common_words}

    # Create bar chart
    plt.figure(figsize=(20, 8))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(words)) for i in range(len(words), 0, -1)]
    bars = plt.bar(words, freqs, color=colors)

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=12)
    plt.title("Top Frequent Words in the Novel", fontsize=20, weight="bold")
    plt.xlabel("Words", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    for bar, freq in zip(bars, freqs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{freq}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            rotation=0,
        )

    st.pyplot(plt)


def display_wordcloud(wordcloud_image_path):
    st.image(wordcloud_image_path, caption="Word Cloud", use_column_width=True)


def display_keyword_distribution(keyword_distribution_image_path):
    st.image(
        keyword_distribution_image_path,
        caption="Keyword Distribution",
        use_column_width=True,
    )


def analyze_and_display(file_path):
    # Load and process the uploaded text
    text = load_text(file_path)

    # Perform basic statistics and text analysis
    basic_statistics(text)
    words = jieba.lcut(text)
    stop_words = load_stop_words()
    words = preprocess_text(text, stop_words)
    words = [word for word in words if len(word) > 1 and word not in stop_words]

    # Word frequency analysis
    word_frequency_analysis(words)
    word_frequency_analysis(words, top_n=50, specific_keywords=[])

    # Generate word cloud and keyword distribution
    wordcloud_image_path = generate_wordcloud(words)
    keyword_distribution_image_path = analyze_keyword_distribution(
        words, [], num_segments=500, top_n=5
    )

    # Display all visualizations
    display_wordcloud(wordcloud_image_path)
    display_keyword_distribution(keyword_distribution_image_path)

    return wordcloud_image_path, keyword_distribution_image_path


# Streamlit file upload and processing
st.title("Novel Text Analyzer")
uploaded_file = st.file_uploader("Upload a novel (txt)", type=["txt"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Process the uploaded file and display visualizations
    wordcloud_image, keyword_distribution_image = analyze_and_display(tmp_file_path)

    st.success("File processed successfully!")
    st.write(f"Word Cloud and Keyword Distribution displayed above.")
else:
    st.info("Please upload a novel text file to get started.")
