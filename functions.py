import re
import jieba
from collections import Counter
from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba.posseg as pseg
import pandas as pd
import chardet
import numpy as np
import io
import wordcloud
import streamlit as st


def load_stop_words(
    file_path=r"C:\Users\User\Documents\GitHub\novel-analyzer\custom_stopwords.txt",
):
    with open(file_path, "r", encoding="utf-8") as file:
        stop_words = set(
            word.strip().lower() for word in file.read().splitlines()
        )  # Convert to lowercase
    return stop_words


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read(10000)  # Read the first 10KB of the file
    result = chardet.detect(raw_data)
    return result["encoding"]


def load_text(file_path):
    encoding = detect_encoding(file_path)
    print(f"Detected file encoding: {encoding}")
    with open(file_path, "r", encoding=encoding, errors="ignore") as file:
        text = file.read()
    return text


# Function for text preprocessing: Tokenization, lowercasing, and stop words removal
def preprocess_text(text, stop_words):
    words = jieba.lcut(text)  # Tokenize the text
    filtered_words = [
        word for word in words if word.lower() not in stop_words
    ]  # Remove stop words (case-insensitive)
    return filtered_words


def basic_statistics(text):
    characters = len(text)
    words = jieba.lcut(text)
    word_count = len(words)
    sentences = re.split(r"[。！？]", text)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    stats = (
        f"字符总数(Characters): {characters}\n"
        f"单词总数(Word Count): {word_count}\n"
        f"句子总数(Sentence Count): {sentence_count}\n"
        f"平均句长(Average Sentence Length): {avg_sentence_length:.2f} 字"
    )
    return stats


def word_frequency_analysis(words, top_n=50, specific_keywords=None):
    # If specific keywords are provided, combine counts for those keywords
    if specific_keywords:
        grouped_freq = Counter()

        for group in specific_keywords:
            if isinstance(group, list):
                # Combine counts for all words in a group
                combined_count = sum(words.count(word) for word in group)
                if (
                    combined_count > 0
                ):  # Only add to the grouped count if there are occurrences
                    grouped_freq["|".join(group)] = combined_count
            else:
                # Single keyword, just count its occurrences
                grouped_freq[group] = words.count(group)

        # Merge grouped frequencies with the general word frequencies
        word_freq = Counter(grouped_freq)

        # Get the top n common words, including grouped ones
        common_words = word_freq.most_common(top_n)

    else:
        # If no specific keywords are provided, count all words
        word_freq = Counter(words)
        common_words = word_freq.most_common(top_n)

    # Print the common words and their frequencies
    for word, freq in common_words:
        print(f"{word}: {freq}")

    # Extract word types (parts of speech) for the common words
    words_pos = {word: pseg.cut(word).__next__().flag for word, _ in common_words}

    # Prepare the data for visualization
    data = pd.DataFrame(
        {
            "词语": [word for word, _ in common_words],
            "频率": [freq for _, freq in common_words],
            "词性": [words_pos[word] for word, _ in common_words],
        }
    )

    # Create a bar chart to visualize word frequencies
    plt.figure(figsize=(20, 8))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    words, freqs = zip(*common_words)

    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(words)) for i in range(len(words), 0, -1)]
    bars = plt.bar(words, freqs, color=colors)

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=12)
    plt.title(
        f"前{top_n}位小说中高频词\nTop {top_n} Most Frequent Words in the Novel",
        fontsize=20,
        weight="bold",
    )
    plt.xlabel("词语 (Words)", fontsize=14)
    plt.ylabel("频率 (Frequency)", fontsize=14)

    for bar, freq in zip(bars, freqs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{freq}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            rotation=45,
        )

    plt.tight_layout()

    # Convert the plot to an image for Streamlit
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format="png")
    image_buffer.seek(0)
    return image_buffer


# In your function, ensure the image buffer is being returned properly
def generate_wordcloud(words):
    word_string = " ".join(words)
    wordcloud_obj = WordCloud(
        font_path="C:\Windows\Fonts\msyh.ttc",  # Make sure the path is correct
        width=800,
        height=400,
        background_color="white",
    ).generate(word_string)

    # Save wordcloud image to a buffer
    image_buffer = io.BytesIO()
    wordcloud_obj.to_image().save(image_buffer, format="PNG")
    image_buffer.seek(0)
    return image_buffer


def analyze_keyword_distribution(word_list, name_ls, num_segments=500, top_n=5):
    # Convert word list to pandas Series
    data = pd.Series(word_list)

    # Initialize DataFrame to hold the histogram data for each keyword (including combined ones)
    hist_df = pd.DataFrame()

    # Set font to one that supports Chinese characters
    font_path = r"C:\Windows\Fonts\simhei.ttf"  # Adjust based on your system
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()

    # If no specific keywords are provided, use the top N frequent words
    if len(name_ls) == 0:
        # Use word frequency analysis to get the top N frequent words
        word_freq = Counter(word_list)
        common_words = word_freq.most_common(top_n)
        words, freqs = zip(*common_words)

        # Generate histograms for the top frequent words
        for word, freq in common_words:
            # Find the positions of the word in the text
            idx = [
                i
                for i, word_in_text in enumerate(data)
                if re.search(r"\b" + re.escape(word) + r"\b", word_in_text)
            ]
            # print(f"Matches for word '{word}': {len(idx)}")

            # Data segmentation: divide the data into 'num_segments' equal bins
            hist_data = np.histogram(idx, bins=num_segments, range=[0, len(data)])[0]
            hist_df[word] = hist_data
            hist_df[word + "_total"] = [len(idx)] * num_segments

        # Plot combined histogram for the top N frequent words
        plt.figure(figsize=(16, 6))
        for word in hist_df.columns:
            if not word.endswith("_total"):  # Ignore total count columns for plotting
                total_count = hist_df[word + "_total"].iloc[0]
                plt.plot(
                    np.linspace(
                        0, 100, num_segments
                    ),  # Change to percentage scale for x-axis
                    hist_df[word],
                    label=f"Word: {word} (Total: {total_count})\n词语: {word} (总数: {total_count})",
                )

        plt.title(
            "Top N Frequent Words Frequency Distribution in Text Segments\n前N个高频词在文本分段中的频率分布",
            fontsize=16,
        )
        plt.xlabel("Progression of Novel (%)\n小说进度 (%)", fontsize=14)
        plt.ylabel("Frequency\n频率", fontsize=14)
        plt.xticks(
            np.linspace(0, 100, 6),  # Set x-axis to show 0%, 20%, 40%, ..., 100%
            [f"{i:.0f}%" for i in np.linspace(0, 100, 6)],
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to buffer for Streamlit
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format="png")
        image_buffer.seek(0)
        return image_buffer
    else:
        # If specific keywords are provided, proceed with the logic for keyword groups
        for keyword_group in name_ls:
            if isinstance(keyword_group, list):
                combined_hist_data = np.zeros(
                    num_segments
                )  # Initialize array for combined counts
                combined_total_count = 0  # To store the total count for the group
                for keyword in keyword_group:
                    # print(f"Searching for individual keyword: {keyword}")
                    # Find the positions of each keyword in the text
                    idx = [
                        i
                        for i, word in enumerate(data)
                        if re.search(r"\b" + re.escape(keyword) + r"\b", word)
                    ]
                    # print(f"Matches for keyword '{keyword}': {len(idx)}")
                    # Data segmentation: divide the data into 'num_segments' equal bins
                    hist_data = np.histogram(
                        idx, bins=num_segments, range=[0, len(data)]
                    )[0]
                    # Add the keyword's histogram data to the combined counts
                    combined_hist_data += hist_data
                    combined_total_count += len(idx)

                # Store the combined histogram data for the group
                hist_df["_".join(keyword_group)] = combined_hist_data
                hist_df["_".join(keyword_group) + "_total"] = [
                    combined_total_count
                ] * num_segments
            else:
                # print(f"Searching for keyword: {keyword_group}")
                # Find the positions of the keyword in the text
                idx = [
                    i
                    for i, word in enumerate(data)
                    if re.search(r"\b" + re.escape(keyword_group) + r"\b", word)
                ]
                # print(f"Matches for keyword '{keyword_group}': {len(idx)}")
                # Data segmentation: divide the data into 'num_segments' equal bins
                hist_data = np.histogram(idx, bins=num_segments, range=[0, len(data)])[
                    0
                ]
                hist_df[keyword_group] = hist_data
                hist_df[keyword_group + "_total"] = [len(idx)] * num_segments

        # Plot combined histogram (all keywords together)
        plt.figure(figsize=(16, 6))
        for keyword in hist_df.columns:
            if not keyword.endswith(
                "_total"
            ):  # Ignore total count columns for plotting
                total_count = hist_df[keyword + "_total"].iloc[0]
                # Plot keyword frequency distribution
                plt.plot(
                    np.linspace(
                        0, 100, num_segments
                    ),  # Change to percentage scale for x-axis
                    hist_df[keyword],
                    label=f"Keyword: {keyword} (Total: {total_count})\n关键词: {keyword} (总数: {total_count})",
                )

        plt.title(
            "Keyword Frequency Distribution in Text Segments\n关键词在文本分段中的频率分布",
            fontsize=16,
        )
        plt.xlabel("Progression of Novel (%)\n小说进度 (%)", fontsize=14)
        plt.ylabel("Frequency\n频率", fontsize=14)
        plt.xticks(
            np.linspace(0, 100, 6),  # Set x-axis to show 0%, 20%, 40%, ..., 100%
            [f"{i:.0f}%" for i in np.linspace(0, 100, 6)],
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to buffer for Streamlit
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format="png")
        image_buffer.seek(0)
        return image_buffer


def main(file_path):
    text = load_text(file_path)
    stop_words = load_stop_words()
    words = jieba.lcut(text)
    words = preprocess_text(text, stop_words)

    # Perform word frequency analysis and generate visualizations
    word_freq_image = word_frequency_analysis(words)
    wordcloud_image = generate_wordcloud(words)
    keyword_distribution_image = analyze_keyword_distribution(words, [])

    # Display the images in Streamlit
    st.image(word_freq_image, caption="Word Frequency Analysis")
    st.image(wordcloud_image, caption="Word Cloud")
    st.image(keyword_distribution_image, caption="Keyword Frequency Distribution")
