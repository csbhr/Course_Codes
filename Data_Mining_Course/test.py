import text_clustering.k_means as k_means
import text_clustering.dbscan as dbscan
import text_clustering.fcm as fcm
import text_clustering.agnes as agnes
import tools.distance_tools as dtools
import tools.file_tools as ftools
import numpy as np

root_path = "D:/WorkSpace/python/project_data/Data_Mining/"
root_path_result = root_path + "result/"
root_path_20_newsgroup = root_path + "20 Newsgroups/"
root_path_AAAI = root_path + "AAAI-14 Accepted Papers - Papers/"
root_path_NYSK = root_path + "NYSK Data Set/"

# x = ftools.read_csv_mat(root_path_20_newsgroup + "feature_vector_after_pca.csv")
# x = ftools.read_csv_mat(root_path_20_newsgroup + "weights_after_pca.csv")
# x = ftools.read_csv_mat(root_path_AAAI + "feature_vector_after_pca.csv")
# x = ftools.read_csv_mat(root_path_AAAI + "weights_after_pca.csv")
# x = ftools.read_csv_mat(root_path_NYSK + "feature_vector_after_pca.csv")
x = ftools.read_csv_mat(root_path_NYSK + "weights_after_pca.csv")

print(x.shape)
