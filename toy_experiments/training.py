import torch
from torch import nn


def train_toy(model, train_images, train_labels, test_images, test_labels, epochs=200, batch_size=32, lr=1e-3, report_every_n=10, weight_decay=0., pos_emb_weight_decay=None):
    if pos_emb_weight_decay is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        params = dict(model.named_parameters())
        pos_emb_params = [params['pos_embedding']]
        other_params = [v for (k,v) in params.items() if k != 'pos_embedding']
        optimizer = torch.optim.Adam([
            {'params': other_params},
            {'params': pos_emb_params, 'weight_decay': pos_emb_weight_decay},
        ], lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)

    # Move to CUDA if available
    cuda = False
    if torch.cuda.is_available():
        cuda = True
        model.cuda()
        criterion.cuda()
        print("Using CUDA.")

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if epoch % report_every_n == 0:
            print(f"Epoch {epoch}: {loss.item()}")

    print(f"Epoch {epoch}: {loss.item()}")
    return test_toy(model, test_images, test_labels, batch_size=batch_size)

def test_toy(model, test_images, test_labels, batch_size=32):
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_images, test_labels), batch_size=batch_size)
    correct = 0
    total = 0

    # Move to CUDA if available
    cuda = False
    if torch.cuda.is_available():
        cuda = True
        model.cuda()
        print("Using CUDA.")

    for batch in test_loader:
        images, labels = batch
        if cuda:
                images = images.cuda()
                labels = labels.cuda()

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).detach().cpu()
        total += labels.shape[0]
    print(f"Accuracy: {correct / total}")
    return correct / total


def single_batch_inference(model, train_images, train_labels):
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_images, train_labels), batch_size=1, shuffle=False)
    for batch in train_loader:
        # optimizer.zero_grad()
        images, labels = batch
        logits = model(images)
        loss = criterion(logits, labels)
        # loss.backward()
        # optimizer.step()
        return loss, logits


