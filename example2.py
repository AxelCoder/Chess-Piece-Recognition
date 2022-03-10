from skimage.io import imshow, imread
from skimage.color import rgb2gray, rgba2rgb
# from skimage import color
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os



# get the filenames of the leaves under the directory “Leaves”
image_path_list = os.listdir("data/Alfil/")


# looking at the first image
i = 2
image_path = image_path_list[i]
image = rgb2gray(imread("data/Alfil/"+image_path))
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
image_path_list = os.listdir("data/Alfil/")
df = pd.DataFrame()

#obtiene los directorios del dataset, la idea es hace lo mismo para cada uno 
dir_image_path_list = os.listdir("data/")
#
for j in range(len(dir_image_path_list)):
    image_path_list = os.listdir("data/"+dir_image_path_list[j]+"/")
    #print (dir_image_path_list[j])
    #print(image_path_list)
    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        #print(image_path)
        # image = rgb2gray(rgba2rgb(imread("data/"+image_path)))
        if(image_path.endswith(".png")):
            image = rgb2gray(rgba2rgb(imread("data/"+dir_image_path_list[j]+"/"+image_path)))
        else:
            image = rgb2gray(imread("data/"+dir_image_path_list[j]+"/"+image_path))
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

        table['label'] = dir_image_path_list[j]

        df = pd.concat([df, table], axis=0)

#print(table)

df.head()

#print(df)


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

print("GradientBoostingClassifier")
# Se aplica el clasiicador, en este caso el clasificador de aumento de gradiente
clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=123)
clf.fit(X_train, y_train)
#print confusion matrix of test set
print(classification_report(clf.predict(X_test), y_test))
#print accuracy score of the test set
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")

# Se aplica el clasiicador, en este caso el clasificador de aumento de gradiente
print("RandomForestClassifier")
clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=123)
clf.fit(X_train, y_train)
#print confusion matrix of test set
print(classification_report(clf.predict(X_test), y_test))
#print accuracy score of the test set
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")

print("StackingClassifier")
estimators = [
('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
('svr', make_pipeline(StandardScaler(),
LinearSVC(random_state=42)))
]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf.fit(X_train, y_train)
print(classification_report(clf.predict(X_test), y_test))
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")

print("ExtraTreesClassifier")
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
print(classification_report(clf.predict(X_test), y_test))
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")


print("DecisionTreesClassifier")
clf = DecisionTreeClassifier(max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
print(classification_report(clf.predict(X_test), y_test))
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")

print("DecisionTreesClassifier")
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(classification_report(clf.predict(X_test), y_test))
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")

print("HistGradientBoostingClassifier")
clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
clf.fit(X_train, y_train)
print(classification_report(clf.predict(X_test), y_test))
print(f"Test Accuracy: {np.mean(clf.predict(X_test) == y_test)*100:.2f}%")