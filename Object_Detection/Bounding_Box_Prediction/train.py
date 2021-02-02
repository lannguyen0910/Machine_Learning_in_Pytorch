import matplotlib.patches as pt
import matplotlib.pyplot as plt

from utils.getter import *
from torch.autograd import Variable


torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    def train(model, optimizer, criterion, train_loader):
        model.train()
        for id, (img, bbox) in enumerate(train_loader):
            optimizer.zero_grad()
            img = Variable(img).view(-1, IMG_SIZE * IMG_SIZE).to(device)
            label = Variable(bbox).view(-1).to(device)

            prediction = model(img).view(-1)
            loss = criterion(label, prediction)

            loss.backward()
            optimizer.step()

        return loss.data

    model = MultiLayerPerception(IMG_SIZE * IMG_SIZE, 4).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss().to(device)

    EPOCHS = 100
    loss_list = []
    for epoch in range(EPOCHS):
        loss = train(model, optimizer, criterion, trainloader)
        loss_list.append(loss)
        print(f'Epoch: {epoch + 1} | Loss: {loss}')

    def plot_data(dataset, bboxes, figsize):
        a = np.random.randint(200)
        fig = plt.figure(figsize=figsize)
        for id, data in enumerate(dataset[a:a + 12]):
            fig.add_subplot(3, 4, id + 1)
            bbox = bboxes[a + id]
            plt.imshow(data, cmap="binary", interpolation='none',
                       origin='lower', extent=[0, IMG_SIZE, 0, IMG_SIZE])
            plt.gca().add_patch(pt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                             ec="r", fc="none", lw=5))
        plt.show()

    def plot_loss():
        plt.plot(range(EPOCHS), loss_list)
        plt.legend()
        plt.show()

    # Evaluate
    testset_tensor = Variable(testset_tensor).view(-1,
                                                   IMG_SIZE*IMG_SIZE).to(device)
    with torch.no_grad():
        prediction = model(testset_tensor)

    predicted_bboxes = prediction.cpu().numpy()

    def plot_eval(testset, testbboxes, figsize):
        fig = plt.figure(figsize=figsize)
        for id, i in enumerate(testset[:10]):
            fig.add_subplot(5, 2, id+1)
            bbox = predicted_bboxes[id]
            iou = iou_compute(bbox, testbboxes[id])
            plt.title("IOU Score " + str(iou))
            plt.imshow(i, cmap="binary", interpolation='none',
                       origin='lower', extent=[0, IMG_SIZE, 0, IMG_SIZE])
            plt.gca().add_patch(pt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                             ec="r", fc="none", lw=5))
        plt.show()

    plot_data(trainset, train_bboxes, (20, 15))
    plot_loss()
    plot_eval(testset, testbboxes, (20, 40))
