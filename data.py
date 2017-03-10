import os, zipfile, urllib.request
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

      string_documents = []
      for raw_document in self.xml.xpath('//content/text()'):
        # Remove everything in parens
        no_parens = re.sub(r'\([^)]*\)', '', raw_document)
        # Remove line breaks and "foo: " prefixes
        merged_lines = re.sub(r'\n([^:]{,20}:)?', ' ', no_parens)
        # Lowercase, remove special chars
        ascii_ = re.sub(r'[^a-z0-9\'\s]+', '', merged_lines.lower())
        string_documents.append(ascii_)

      self.permutation = np.random.permutation(len(string_documents))
      string_documents = np.asarray(string_documents)[self.permutation]

      self.representation = CountVectorizer(stop_words='english',
        max_features=vocab_size-1, token_pattern='(?u)\\b\w+[\w\']*\\b')
      self.representation.fit(string_documents[:1585])

      self.UNKNOWN = 0
      docs = [[self.representation.vocabulary_.get(word, self.UNKNOWN - 1) + 1
          for word in doc.split()] for doc in string_documents]
      self._x_train = np.copy(docs[:1585])
      self.x_valid, self.x_valid_l = self._toNumpy(docs[1585:1835])
      self.x_test, self.x_test_l = self._toNumpy(docs[1835:])

  def _toNumpy(self, array):
    length = np.array([len(a) for a in array])
    mask = np.arange(length.max()) < length[:,None]
    nparray = np.zeros(mask.shape)
    nparray[mask] = np.concatenate(array)
    return nparray, length

  def vocabulary(self):
    return self.representation.vocabulary_

class TedDataWithLabels(TedData):
  def __init__(self, vocab_size):
    TedData.__init__(self, vocab_size)
    labels = [
      (1 if "technology" in raw_category else 0) +
      (2 if "entertainment" in raw_category else 0) +
      (4 if "design" in raw_category else 0)
      for raw_category in self.xml.xpath('//keywords/text()')]
    del self.xml
    labels = np.asarray(labels)[self.permutation]
    self._y_train = np.copy(labels[:1585])
    self.y_valid = labels[1585:1835]
    self.y_test = labels[1835:]

  def training_batches(self, batch_size):
    docs, labels = shuffle(self._x_train, self._y_train)
    for i in range(0, 1585, batch_size):
      a, l = self._toNumpy(docs[i:i+batch_size])
      yield a, l, labels[i:i+batch_size]


class TedDataSeq(TedData):
  def __init__(self, vocab_size):
    TedData.__init__(self, vocab_size-3)
    del self.xml

    self.START = vocab_size - 3
    self.END = vocab_size - 2
    self.BLANK = vocab_size - 1

    self.x_valid, self.y_valid = self._pad(self.x_valid[1585:1835])
    self.x_test, self.y_test = self._pad(self.x_test[1835:])

  def _pad(self, documents):
    length = max(map(len, documents))
    batch = np.array([[self.START]+document+[self.END] + [self.BLANK]*(length-len(document)) for document in documents])
    return batch[:,:-1], batch[:,1:]

  def training_batches(self, batch_size):
    docs = shuffle(self._x_train)
    docs[np.argsort(map(len, docs), kind='mergesort')]
    # docs are now sorted but random within the same length
    for i in range(0, 1585, batch_size):
      batch = docs[i:i+batch_size]
      yield self._pad(batch, max(map(len, batch)))

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
