import re
import lxml.etree
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def ted(vocab_size):
  if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

  with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
    raw_categories = doc.xpath('//keywords/text()')
    raw_documents = doc.xpath('//content/text()')
    del doc

    documents = []
    categories = []
    for (raw_document, raw_category) in zip(raw_documents, raw_categories):
        # Remove everything in parens
        no_parens = re.sub(r'\([^)]*\)', '', raw_document)
        # Remove line breaks and "foo: " prefixes
        merged_lines = re.sub(r'\n([^:]{,20}:)?', ' ', no_parens)
        # Lowercase, remove special chars
        ascii_ = re.sub(r'[^a-z0-0\.]+', ' ', merged_lines.lower())
        documents.append(ascii_)

        # ["ooo", "Too", "oEo", "TEo", "ooD", "ToD", "oED", "TED"]
        categories.append(
          (1 if "technology" in raw_category else 0) +
          (2 if "entertainment" in raw_category else 0) +
          (4 if "design" in raw_category else 0))
    del raw_documents
    del raw_categories

    cv = CountVectorizer(stop_words='english', max_features=vocab_size)
    bags_of_words = cv.fit_transform(documents)

    return {"documents": bags_of_words.todense(), "categories": categories, "vocab": cv.vocabulary_}


def glove(embedding_dim):
  if not os.path.isfile('glove.6B.zip'):
    urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")

  embedding = {}
  with zipfile.ZipFile('glove.6B.zip', 'r') as z:
    lines = z.open('glove.6B.'+str(embedding_dim)+'d.txt', 'r')
    for line in lines:
        items = line.decode("utf-8").strip().split(' ')
        assert len(items) == 51
        word = items[0]
        vec = [float(i) for i in items[1:]]
        embedding[word] = vec
    return embedding

