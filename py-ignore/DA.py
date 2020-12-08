import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv('trans_insurance.csv')

def get_corr1():
    df = pd.read_csv('trans_insurance.csv')
    '''
    print(df.corr()['charges'].sort_values())
    f, ax = plt.subplots(figsize=(10, 10))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240, 10, as_cmap=True),
                square=True, ax=ax)
    '''
    f, ax = plt.subplots(figsize=(10, 10))
    plt.title('Correlation')
    # Draw the heatmap using seaborn
    sns.heatmap(df.astype(float).corr(), linewidths=0.5, vmax=1.0, square=True, annot=True, ax=ax)


    #sns.lmplot(x="bmi", y="charges", hue="smoker", data=df, palette='magma', size=8)
    plt.show()
    #plt.savefig("plot/corr2/p3", dpi=500)
    plt.clf()

def main():
    get_corr1()

    inertia = []
    pca = PCA(n_components=2)
    dfp = pca.fit_transform(df)
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=1).fit(dfp)
        inertia.append(np.sqrt(kmeans.inertia_))

    plt.plot(range(1, 8), inertia, marker='s')
    plt.title('PCA')
    #plt.xlabel('X')
    #plt.ylabel('Y');
    plt.show()
    #plt.savefig("plot/corr2/p2", dpi=500)
    plt.clf()


    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(dfp)
    LABEL_COLOR_MAP = {0: 'r',
                       1: 'g',
                       2: 'b',
                       3: 'y',
                       4: 'c'}

    label_color = [LABEL_COLOR_MAP[l] for l in clusters]
    plt.figure(figsize=(10, 10))
    plt.scatter(dfp[:, 0], dfp[:, 1], c=label_color, alpha=0.9)
    plt.title('Cluster')
    #plt.savefig("plot/corr2/p4", dpi=500)
    plt.show()


if __name__=='__main__':
    main()