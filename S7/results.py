import matplotlib.pyplot as plt
import numpy as np

def plot_misclassified_images(img_data, classes, img_name):
  figure = plt.figure(figsize=(10, 10))
  
  num_of_images = len(img_data)
  for index in range(1, num_of_images + 1):
      img = img_data[index-1]["img"] / 2 + 0.5     # unnormalize
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(np.transpose(img, (1, 2, 0)))
      plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))
  
  plt.tight_layout()
  plt.savefig(img_name)

def plot_graph(data, metric):
    fig = plt.figure(figsize=(7, 7))
    
    plt.title(f'Validation %s' % (metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.plot(data)
    plt.show()
    
    fig.savefig(f'val_%s_change.png' % (metric.lower()))