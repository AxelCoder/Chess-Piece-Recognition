from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu # Librería para otsu segmentation
from skimage import feature # Librería para canny
from skimage.morphology import closing
from matplotlib import pyplot as plt
import os



# Obtenemos los nombres de archivos de las piezas en el directorio data
image_path_list = os.listdir("data")

# Tomamos la segunda imagen
i = 1
image_path = image_path_list[i]
image = rgb2gray(imread("data/"+image_path))

# Preprocesando imagen usando el método Otsu segmentation
binary = image < threshold_otsu(image)
binary = closing(binary)
imshow(binary)
plt.savefig("imagen-Otsu_segmentation")


# Preprocesando imagen usando el método Canny edge detector
edges2 = feature.canny(image, sigma=3)
imshow(edges2)
plt.savefig("imagen-Canny_edge_dtector")

