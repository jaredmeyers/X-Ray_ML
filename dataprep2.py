import pickle, tqdm, os, pandas as pd, numpy as np
from pydicom import dcmread
from multiprocessing import Pool
from tqdm import tqdm
from skimage.transform import resize

def get_data(name_and_label):
    try:
        name, label = name_and_label[0]+'.dicom', name_and_label[1]
        ds = dcmread(name)
        arr = ds.pixel_array
        arr = resize(arr, (32,32), anti_aliasing = True)
        #print(arr.shape)
        '''import matplotlib.pyplot as plt
        plt.imshow(arr, cmap= 'gray')
        plt.show()'''
        return (arr, label)
    except:
        pass



if __name__ == '__main__':
    
    os.chdir("C:\\Users\\Administrator\\Documents\\School\Machine-Learning\\data")

    df = pd.read_csv("C:\\Users\\Administrator\\Documents\\School\\Machine-Learning\\data\\xray\\train.csv")
    dicoms = os.listdir("C:\\Users\\Administrator\\Documents\\School\\Machine-Learning\\data\\xray\\train")
    print('Indexing...')
    indexed = []
    for i in tqdm(range(len(dicoms))):    
        indexed.append((df['image_id'][i], int(df['class_id'][i])))
    
    print('Creating Arrays...')
    os.chdir("C:\\Users\\Administrator\\Documents\\School\\Machine-Learning\\data\\xray\\train")
    processes = 2
    p = Pool(processes = processes)
    results = list(tqdm(p.imap(get_data, indexed), total = len(indexed)))
    p.close()
    p.join()
    with open('pickle.data', 'wb') as f:
        pickle.dump(results, f)
        f.close()
    #print(results)


#print(get_data(('000ae00eb3942d27e0b97903dd563a6e',7)))
