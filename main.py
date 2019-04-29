import sys
import os
import numpy as np
import matplotlib
"""
Workaround to solve bug using matplotlib. See:
https://stackoverflow.com/questions/49367013/pipenv-install-matplotlib
"""
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

K_DIMENSION = [1, 2, 3, 6, 10, 20, 30, 50]

class PCA:
  def __init__(self, training_set):
    self.data = training_set
    self.train()

  def train(self):
    data = np.array(list(map(lambda x: x['data'], self.data))).T
    mean = np.mean(data, axis=1)
    mean = mean.reshape(mean.size, 1)

    X = data - mean
    self.mean = mean
    self.normalized_data = X

    eig_vals, eig_vecs = np.linalg.eig(X.dot(X.T))
    eig_vals_sorted = np.sort(eig_vals)
    eig_vecs_sorted = eig_vecs[:, eig_vals.argsort()]
    eig_vals_sorted = np.flip(eig_vals_sorted)
    eig_vecs_sorted = np.flip(eig_vecs_sorted, 0).T

    self.eig_vals = eig_vals_sorted
    self.eig_vecs = eig_vecs_sorted

  def set_k(self, k):
    U = self.eig_vecs[:,:k]
    self.U = U
    self.Y = U.T.dot(self.normalized_data)

  def classify(self, img_data):
    assert len(self.U) > 0

    img = np.array(img_data)
    img = img.reshape(img.size, 1)
    projection = self.U.T.dot(img - self.mean)

    all_dist = []
    for i in range(len(self.data)):
      all_dist.append(np.linalg.norm(projection - self.Y[:, i:i+1]))

    idx = np.argmin(all_dist)
    return self.data[idx]['label']


def plot_accuracies(accuracies):
  """
  Render the line chart. `matplotlib` conflicts with pipenv 😞. Check:
  https://matplotlib.org/faq/osx_framework.html
  """
  plt.plot(K_DIMENSION, list(map(lambda x : x * 100, accuracies)), 'r-o')
  plt.axis([0, max(K_DIMENSION), 0, 100])
  plt.xlabel('K (reduced dimension)')
  plt.ylabel('Accuracy (%)')
  plt.show()

def to_accuracy_by_samples(training_set, test_set):
  """
  Closure to keep training and test set, and calculates eigenvectors only once.
  Returns the mapping function to calculate accuracy rate of given K dimension.
  """
  pca = PCA(training_set)

  def calcAccuracy(k):
    pca.set_k(k)
    correct_res = 0
    for test in test_set:
      label = pca.classify(test['data'])
      if (label == test['label']):
        correct_res += 1
    return correct_res / len(test_set)

  return calcAccuracy

def read_and_parse_file(folder_path):
  """
  Read and parse all files from the subdirectory under given folder_path.
  And split them into training set and test set using the definition below:
  `file name is (2, 6, 8, 10)`
  """
  TEST_SET_DEF = { 2: True, 6: True, 8: True, 10: True }
  training_set, test_set = [], []

  paths = os.listdir(folder_path)
  path_names = [folder_path + '/' + path_name for path_name in paths if not path_name.startswith('.')]

  for path_name in path_names:
    files = os.listdir(path_name)
    for file_name in files:
      with open(path_name + '/' + file_name, 'rb') as f: # 'rb' => 'b': binary mode
        data = { 'label': path_name, 'data': read_pgm(f) }
        f_name = int(file_name.split('.')[0])
        test_set.append(data) if f_name in TEST_SET_DEF else training_set.append(data)

  return training_set, test_set

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
      for y in range(width):
        raster.append(np.frombuffer(pgmf.read(1), dtype=np.int8).item())
    return raster

if __name__ == '__main__':
    folder_path = sys.argv[1]

    # Parse data
    training_set, test_set = read_and_parse_file(folder_path)

    # Get mapped accuracy
    toAccuracy = to_accuracy_by_samples(training_set, test_set)
    accuracies = list(map(toAccuracy, K_DIMENSION))

    # Plot the accuracy with nice line chart
    plot_accuracies(accuracies)