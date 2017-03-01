import os, zipfile
import re, lxml.etree
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

class TedData(object):

  def __init__(self, vocab_size):
    if not os.path.isfile('ted_en-20160408.zip'):
      urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
      self.xml = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

      self.string_documents = []
      for raw_document in self.xml.xpath('//content/text()'):
        # Remove everything in parens
        no_parens = re.sub(r'\([^)]*\)', '', raw_document)
        # Remove line breaks and "foo: " prefixes
        merged_lines = re.sub(r'\n([^:]{,20}:)?', ' ', no_parens)
        # Lowercase, remove special chars
        ascii_ = re.sub(r'[^a-z0-9\'\s]+', '', merged_lines.lower())
        self.string_documents.append(ascii_)

      self.representation = CountVectorizer(stop_words='english',
        max_features=vocab_size, token_pattern='(?u)\\b\w+[\w\']*\\b')
      self.representation.fit(self.string_documents[:1585])

  def vocabulary(self):
    return self.representation.vocabulary_

class TedDataWithLabels(TedData):
  def __init__(self, vocab_size):
    TedData.__init__(self, vocab_size)
    self.labels = [
      (1 if "technology" in raw_category else 0) +
      (2 if "entertainment" in raw_category else 0) +
      (4 if "design" in raw_category else 0)
      for raw_category in self.xml.xpath('//keywords/text()')]
    del self.xml
    documents = self.representation.transform(self.string_documents).todense()
    self._x_train, self._y_train = np.copy(documents[:1585]), np.copy(self.labels[:1585])
    self.x_valid, self.y_valid = documents[1585:1835], self.labels[1585:1835]
    self.x_test, self.y_test = documents[1835:], self.labels[1835:]

  def training_batches(self, batch_size):
    docs, labels = shuffle(self._x_train, self._y_train)
    for i in range(0, 1585, batch_size):
      yield docs[i:i+batch_size], labels[i:i+batch_size]


class TedDataSeq(TedData):
  def __init__(self, vocab_size):
    TedData.__init__(self, vocab_size)
    del self.xml

    self.UNKNOWN = len(self.vocabulary)
    self.START = self.UNKNOWN + 1
    self.END = self.START + 1
    self.BLANK = self.END + 1
    documents = [[self.representation.vocabulary_.get(word, self.UNKNOWN)
        for word in document.split()]
      for document in self.string_documents]
    self._x_train = np.copy(documents[:1585])
    self.x_valid, self.y_valid = self.pad(documents[1585:1835])
    self.x_test, self.y_test = self.pad(documents[1835:])

  def _pad(self, documents):
    length = max(map(len, documents))
    return np.array([np.pad([self.START]+document+[SELF.END], length+2, 'constant', constant_values=self.BLANK)
      for document in documents]),
           np.array([np.pad(documents+[SELF.END], length+2, 'constant', constant_values=self.BLANK)
      for document in documents])

  def training_batches(self, batch_size):
    docs = shuffle(self._x_train)
    docs[np.argsort(map(len, docs), kind='mergesort')]
    # docs are now sorted but random within the same length
    for i in range(0, 1585, batch_size):
      batch = docs[i:i+batch_size]
      padded = self.pad(batch, max(map(len, batch)))
      xs = [entry[:-1] for entry in batch]
      ys = [entry[1:] for entry in batch]
      yield x, y

def Glove(dims):
  if not os.path.isfile('glove.6B.zip'):
    urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
  with zipfile.ZipFile('glove.6B.zip', 'r') as z:
    file = z.open('glove.6B.'+str(dims)+'d.txt', 'r')
    embedding = {}
    for line in file:
      items = line.decode("utf-8").strip().split(' ')
      assert len(items) == 51
      word = items[0]
      vec = [float(i) for i in items[1:]]
      embedding[word] = vec
    return embedding
