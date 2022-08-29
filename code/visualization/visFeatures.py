import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd

maker=['o','v','^','s','p','*','<','>','D','d','h','H']#设置散点形状
colors = ['black','tomato','cyan','blue', 'lime', 'r', 'violet','m','yellow','peru','olivedrab','hotpink']#设置散点颜色

Label_Com = ['1', '2', '3', '4', '5',
             'T-3', 'S-4', 'T-4','S-5','T-5', 'S-6', 'T-6', 'S-7','T-7','S-8', 'T-8','S-9','T-9',
             'S-10','T-10','S-11', 'T-11', 'S-12','T-12'] ##图例名称

### 设置字体格式
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 32,
}

def visual(X):
    tsne = manifold.TSNE(n_components=2,init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    #'''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return  X_norm

def plot_with_labels(S_lowDWeights,Trure_labels,name, class_num):
    plt.cla()#清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels=Trure_labels.reshape((-1,1))
    
    S_data=np.hstack((S_lowDWeights,True_labels))
    S_data=pd.DataFrame({'x':S_data[:,0],'y':S_data[:,1],'label':S_data[:,2]})



    for index in range(class_num):
        X= S_data.loc[S_data['label'] == index]['x']
        Y=S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X,Y,cmap='brg', s=100, marker=maker[0], c='', edgecolors=colors[index],alpha=0.65)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name,fontsize=32,fontweight='normal',pad=20)
    
def load_data(filepath):
    with open(filepath) as file_obj:
        features = json.load(file_obj)
    feat = []
    cls = []
    keys = []
    for (key,value) in features.items():
        keys.append(key)
        feat.append(value['features'][0])
        cls.append(value['class'][0])
    feat = np.array(feat)
    cls = np.array(cls)
    keys = np.array(keys)
    return feat, cls, keys

def similarity(feat, cls, key):
    X = feat
    X_train = feat
    dist = np.reshape(np.sum(X**2,axis=1),(X.shape[0],1))+ np.sum(X_train**2,axis=1)-2*X.dot(X_train.T)
    return dist

def get_label_map():
    label_map = {}
#     labels = ['soma', 'neurite', 'glial']
    labels = ['glial']
    for label in labels:
        if label == 'soma':
            annotation_file = '/code/labels/glial.txt'
            label_id = 0
        elif label == 'neurite':
            annotation_file = '/code/labels/glial.txt'
            label_id = 1
        elif label == 'glial':
            annotation_file = '/code/labels/glial.txt'
            label_id = 2
        for line in open(annotation_file):
            line = line.split()
            for i in line:
                label_map[i] = label_id
    return label_map
        
    
        
    
if __name__ == "__main__":
    filename = './train_features.json'
    image_path = './features1.png'
    
    feat, cls, key = load_data(filename)
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    plot_with_labels(visual(feat),cls,'(a)',5)


    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                wspace=0.1, hspace=0.15)
    plt.legend(scatterpoints=1,labels = Label_Com, loc='best',labelspacing=0.4,columnspacing=0.4,markerscale=2,bbox_to_anchor=(0.9, 0),ncol=12,prop=font1,handletextpad=0.1)

    plt.savefig(image_path, format='png',dpi=300, bbox_inches='tight')
