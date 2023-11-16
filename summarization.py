from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import networkx as nx
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge import Rouge


# Sample Audio Transcription
transcription = """
In the heart of Chesterfield, where cobblestone lanes wound like sleepy serpents through rows of old townhouses, there was a particular residence that stood out. Not because it was grander or more ornate, but because of the peculiar antics of Mr. Pemberton's cat, Whiskers.

Mr. Pemberton was an elderly man with a tuft of white hair that always looked windswept, even on the calmest days. He spent his days reading books and tending to his garden, which was famous for its rare blooms. But it was Whiskers that drew the neighborhood’s attention.

Every evening, as the town's antique clock tower chimed six, Whiskers would stroll out of Mr. Pemberton's front door with a small basket in his mouth. He would visit Mrs. Appleby, the baker two doors down, where she'd place a fresh croissant in his basket. Then he'd saunter over to Mr. Greaves, the butcher, who'd drop in a tiny, cooked sausage. Finally, Whiskers would swing by the house of Miss Penelope, who would add a sprig of rosemary to the collection.

The journey never varied, and neither did the contents of the basket. By the end of his route, Whiskers would return to Mr. Pemberton, who sat waiting on his porch, and together, they'd share their evening meal.

The spectacle became such a routine that tourists began showing up. People would take photographs and videos of the "wonder cat of Chesterfield." Children, their faces pressed to the ground, would watch in awe as the small feline made his way with purpose and poise.

One day, a journalist named Clara decided she would uncover the mystery behind Whiskers' ritual. She approached Mr. Pemberton with a notebook and a head full of questions.

"Mr. Pemberton," Clara began, "how did this all start? This routine with Whiskers?"

Mr. Pemberton chuckled, a warm sound that seemed to shake his entire being. "Ah, dear Clara, it's a simple tale. Many years ago, when Whiskers was but a kitten, he would follow me as I ran my errands. Mrs. Appleby would give him a croissant as a treat, Mr. Greaves a piece of sausage, and Miss Penelope, my dear sister, always added a sprig of rosemary for aroma."

"Over the years, I grew frail and found it hard to walk, but Whiskers continued our tradition. The town's folks began contributing, and now, it's his little ritual," he explained, gazing fondly at the cat, who was, at that moment, receiving his daily croissant.

Intrigued, Clara wrote a heartwarming piece about the town's unity, the dedication of a cat, and the enduring spirit of routines that bring joy. The article was published in a major magazine, and overnight, Chesterfield, with its winding lanes and old-world charm, became even more famous, all thanks to a cat named Whiskers.

Years passed, and while things around them changed, Mr. Pemberton's porch and Whiskers' evening quest remained a constant, comforting presence. The tale of their bond, a testament to routines, love, and community, would be told and retold, long after the cobblestone streets had seen the last of both the old man and his loyal feline friend.
"""

# Tokenize the transcription into sentences
sentences = sent_tokenize(transcription)


# TF-IDF Summarization
def tfidf_summary(sentences, n=3):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    cosine_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(cosine_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [ranked_sentences[i][1] for i in range(n)]
    return ' '.join(top_n_sentences)


def text_rank_or_lex_rank_summary(text, algorithm="textrank", n=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if algorithm == "textrank":
        summarizer = TextRankSummarizer()
    else:
        summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, n)
    return ' '.join([sentence.__str__() for sentence in summary])


print("TF-IDF Summary:")
print(tfidf_summary(sentences, 3))
print("\nTextRank Summary:")
print(text_rank_or_lex_rank_summary(transcription, "textrank", 3))
print("\nLexRank Summary:")
print(text_rank_or_lex_rank_summary(transcription, "lexrank", 3))


# Assuming this is the reference (human-written) summary
reference_summary = """
Whiskers, Mr. Pemberton's cat, has a daily routine of collecting food items from various townsfolk in Chesterfield. This spectacle has become a tourist attraction. Clara, a journalist, learns from Mr. Pemberton that this routine started when Whiskers was a kitten and has continued even as Mr. Pemberton grew older. The tale of Whiskers and Mr. Pemberton highlights community bonding.
"""

def evaluate_summary(generated_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores[0]

tfidf_eval = evaluate_summary(tfidf_summary(sentences, 3), reference_summary)
textrank_eval = evaluate_summary(text_rank_or_lex_rank_summary(transcription, "textrank", 3), reference_summary)
lexrank_eval = evaluate_summary(text_rank_or_lex_rank_summary(transcription, "lexrank", 3), reference_summary)

print("\nTF-IDF Evaluation:")
print(f"ROUGE-1: {tfidf_eval['rouge-1']}")
print(f"ROUGE-2: {tfidf_eval['rouge-2']}")
print(f"ROUGE-L: {tfidf_eval['rouge-l']}")

print("\nTextRank Evaluation:")
print(f"ROUGE-1: {textrank_eval['rouge-1']}")
print(f"ROUGE-2: {textrank_eval['rouge-2']}")
print(f"ROUGE-L: {textrank_eval['rouge-l']}")

print("\nLexRank Evaluation:")
print(f"ROUGE-1: {lexrank_eval['rouge-1']}")
print(f"ROUGE-2: {lexrank_eval['rouge-2']}")
print(f"ROUGE-L: {lexrank_eval['rouge-l']}")
