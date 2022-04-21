from dataset import *
from model import *

import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans

root = "/media/q/data/lihongyi2020data/hw9/"

from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        import torch
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')
def main():
    # load model
    model = AE().cuda()
    model.load_state_dict(torch.load(root + 'ckpt/last_checkpoint.pth'))
    model.eval()

    # 準備 data
    trainX = np.load(root + 'trainX.npy')

    # 預測答案
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    # 將預測結果存檔，上傳 kaggle
    save_prediction(pred, 'prediction.csv')

    # 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
    # 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
    save_prediction(invert(pred), 'prediction_invert.csv')


if __name__ == '__main__':
    main()