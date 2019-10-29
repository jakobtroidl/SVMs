import mnist
import numpy as np

# load training data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
# load test data
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# classes to be classified
class1 = 7
class2 = 2

# get indices of 7 & 2 images
my_indices = np.where(train_labels[0] == class1 or train_labels[0] == class2)

# filter images & labels based on the extracted indices
my_images = train_images[my_indices[0], :, :]
my_labels = train_labels[my_indices[0], :, :]


