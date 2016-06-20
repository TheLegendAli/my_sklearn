# This script does kmeans clustering

import numpy as np
import cPickle as pickle
import h5py
import pdb
from sklearn.cluster import KMeans


path = '/media/work/unsupervised-part-learning2/dictionary_pascal_2010_sky_vgg16_pool4_nowarp.mat'
#path = 'dictionary_pascal_2010_grass_vgg16_pool3_nowarp.mat'
f = h5py.File(path)
feat_set = np.array(f['feat_set'])
loc_set = np.array(f['loc_set'])
img_set = np.array(f['img_set'])

K = 45
layer_name = 'pool4'

#pdb.set_trace()
# L2 normalization as preprocessing
feat_square = np.square(feat_set)
feat_norm = np.sqrt(np.sum(feat_square, axis = 0))
normalized_feat_set = feat_set/feat_norm

# # L1 normalization
# feat_norm = sum(feat_set, 1)
#change this
# normalized_feat_set = feat_set/feat_norm

#run k++ means
cluster = KMeans(init='k-means++', n_clusters=K, verbose=1).fit(normalized_feat_set)
#pdb.set_trace()
