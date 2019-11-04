import mnist
import numpy as np
# import skimage

trainImages = mnist.train_images()
trainLabels = mnist.train_labels()
testImages = mnist.test_images()
testLabels = mnist.test_labels()

classes = [2, 7]

indexes = list(filter(lambda x: trainLabels[x] in classes, range(len(trainLabels))))

selectedImages = trainImages[indexes, :, :]
selectedLabels = trainLabels[indexes]

subsetSize = 500

subsetTauImages = selectedImages[0:subsetSize]
subsetTauLabels = selectedLabels[0:subsetSize]

# Filled area
subsetTauFilledArea = []
for image in subsetTauImages:
    zeroOnes = np.sign(image)
    subsetTauFilledArea.append(sum(sum(zeroOnes))/(len(zeroOnes)*len(zeroOnes[0])))
