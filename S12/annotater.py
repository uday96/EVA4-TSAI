import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class Annotater(object):
	"""Finds the template boxes by clustering the bboxes"""
	def __init__(self, in_path, out_path):
		super(Annotater, self).__init__()
		annotations = self._parse_annotations(in_path)
		self._parse_and_save(annotations, out_path)

	def _parse_annotations(self, path):
		with open(path, 'r') as f:
			annotations = json.load(f)
		
		classes = {}
		for val in annotations["categories"]:
			classes[val["id"]] = val["name"]

		data = {}
		for val in annotations["images"]:
			data[val["id"]] = {
				'name': val["file_name"],
				'height': val["height"], 
				'width': val["width"],
			}
		for val in annotations["annotations"]:
			data[int(val["image_id"])]["class"] = classes[val["category_id"]]
			data[int(val["image_id"])]["bbox"] = val["bbox"]

		return data

	def _parse_and_save(self, annotations, path):
		df_data = {
			"img_name": [],
			"class": [],
			"img_h": [],
			"img_w": [],
			"bbox_x": [],
			"bbox_y": [],
			"bbox_h": [],
			"bbox_w": [],
			"bbox_scaled_x": [],
			"bbox_scaled_y": [],
			"bbox_scaled_w": [],
			"bbox_scaled_h": [],
		}
		bboxes = []
		for val in annotations.values():
			df_data["img_name"].append(val["name"])
			df_data["class"].append(val["class"])
			df_data["img_h"].append(val["height"])
			df_data["img_w"].append(val["width"])
			df_data["bbox_x"].append(val["bbox"][0])
			df_data["bbox_y"].append(val["bbox"][1])
			df_data["bbox_h"].append(val["bbox"][2])
			df_data["bbox_w"].append(val["bbox"][3])
			df_data["bbox_scaled_x"].append(val["bbox"][0]/val["width"])
			df_data["bbox_scaled_y"].append(val["bbox"][1]/val["height"])
			df_data["bbox_scaled_w"].append(val["bbox"][2]/val["width"])
			df_data["bbox_scaled_h"].append(val["bbox"][3]/val["height"])
			bboxes.append((df_data["bbox_scaled_w"][-1],
						df_data["bbox_scaled_h"][-1]))

		df = pd.DataFrame(df_data)
		df.to_csv(path)
		print("Saved image bbox data at: %s\n" % path)
		print("Showing the first few rows of generated bbox data:\n")
		print(df.head())

		self.bboxes = np.array(bboxes)
		self.log_bboxes = self.bboxes.copy()
		self.log_bboxes = np.log(self.log_bboxes)

	def show_bboxes(self):
		fig, axs = plt.subplots(1, 2, figsize=(12, 4))
		fig.suptitle("Bounding Boxes")
		axs[0].scatter(self.bboxes[:,0], self.bboxes[:,1])
		axs[0].set_xlabel('w')
		axs[0].set_ylabel('h')
		axs[1].scatter(self.log_bboxes[:,0], self.log_bboxes[:,1])
		axs[1].set_xlabel('log(w)')
		axs[1].set_ylabel('log(h)')
		plt.show()
		fig.tight_layout(pad=3.0)
		fig.savefig("bboxes.png")

	def try_cluster(self):
		X = self.log_bboxes
		wcss = []
		for i in range(1, 11):
			kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
							n_init=10, random_state=0)
			kmeans.fit(X)
			wcss.append(kmeans.inertia_)
		plt.plot(range(1, 11), wcss)
		plt.title('Elbow Method')
		plt.xlabel('Number of clusters')
		plt.ylabel('WCSS')
		plt.savefig("kmeans_elbow.png")
		plt.show()

	def fit(self, k):
		X = self.log_bboxes
		kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300,
						n_init=10, random_state=0)
		pred_y = kmeans.fit_predict(X)
		self.centroids = kmeans.cluster_centers_
		plt.scatter(X[:,0], X[:,1])
		plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
					s=300, c='red')
		plt.title("K=%s Clustered Bboxes" % k)
		plt.savefig("k%s_clustered_bboxes.png" % k)
		plt.show()

