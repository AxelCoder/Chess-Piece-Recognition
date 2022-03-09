from skimage.io import imshow, imread
from skimage.color import rgb2gray, rgba2rgb
# from skimage import color
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os



# get the filenames of the leaves under the directory “Leaves”
image_path_list = os.listdir("data")


# looking at the first image
i = 2
image_path = image_path_list[i]
image = rgb2gray(imread("data/"+image_path))
# image = rgb2gray(rgba2rgb(imread("data/"+image_path)))
imshow(image)

# preprocesamos la imagen binarizándola primero usando el método de Otsu
binary = image < threshold_otsu(image)
binary = closing(binary)
imshow(binary)

# etiquetamos la imagen preprocesada
label_img = label(binary)
imshow(label_img)
plt.show()


# extraemos las características de la imagen usando las propiedades de la región
image_path_list = os.listdir("data")
df = pd.DataFrame()

for i in range(len(image_path_list)):
    image_path = image_path_list[i]
    # image = rgb2gray(rgba2rgb(imread("data/"+image_path)))
    image = rgb2gray(imread("data/"+image_path))
    binary = image < threshold_otsu(image)
    binary = closing(binary)
    label_img = label(binary)
  

    # Calculamos 4 características a partir de las propiedades de la región.
    #   1. inertia_tensor — Esto se relaciona con la rotación del segmento alrededor de su masa.
    #   2. minor_axis_length — longitud del eje más corto del segmento.
    #   3. solidity — relación entre el área del casco convexo y el área de la imagen binaria.
    #   4. eccentricity — relación de la distancia focal (distancia entre los puntos focales) sobre la longitud del eje principal.
    
    table = pd.DataFrame(regionprops_table(label_img, image,
                                           ['convex_area', 'area',
                                            'eccentricity', 'extent',                   
                                            'inertia_tensor',
                                            'major_axis_length', 
                                            'minor_axis_length',
                                            'solidity']))
    table['convex_ratio'] = table['area']/table['convex_area']

    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []

    for prop in regionprops(label_img):
        min_row, min_col, max_row, max_col = prop.bbox
        img = image[min_row:max_row,min_col:max_col]
        real_images += [img]
        mean += [np.mean(img)]
        std += [np.std(img)]
        percent25 += [np.percentile(img, 25)] 
        percent75 += [np.percentile(img, 75)]

    table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']

    table['label'] = image_path[5]

    df = pd.concat([df, table], axis=0)

df.head()


# Implementación del aprendizaje automático.
# X = df.drop(columns=['label', 'image', 'real_images'])
X = df.drop(columns=['label', 'real_images'])

#features
X = X[['iqr','75th Percentile','inertia_tensor-1-1',
       'std_intensity','mean_intensity','25th Percentile',
       'minor_axis_length', 'solidity', 'eccentricity']]
#target
y = df['label']
columns = X.columns
#train-test-split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=123, stratify=y)


# Se aplica el clasiicador, en este caso el clasificador de aumento de gradiente
clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=123)
clf.fit(X_train, y_train)
#print confusion matrix of test set
print(classification_report(clf.predict(X_test), y_test))
#print accuracy score of the test set
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")