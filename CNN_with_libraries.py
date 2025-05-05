#Οι βιβλιοθήκες που χρησιμοποιήθηκαν
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import random

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  #Random flip
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

batch_size = 32

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# To CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #Συνελικτικά επίπεδα
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #Max pooling
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #Συνδεδεμένα Επίπεδα
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5) #Drop out για την αποφυγή του overfitting
        self.relu = nn.ReLU() # Συνάρτηση ενεργοποίησης στην έξοδο

        #Εμπρόσθιο πέρασμα
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten for the fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Σε περίπτωση που τρέξει σε gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss() #cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam optimizer

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

#Εκπαίδευση του μοντέλου
num_epochs = 10
start_time = time.time()#Ξεκίνημα χρονομέτρησης
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Ξεκίνημα του χρονομέτρου κάθε εποχής
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Μηδενίζω τις παραμέτρους των παραγώγων
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Ανανέωση βαρών

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(trainloader))
    train_accuracies.append(100 * correct / total)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(testloader))
    test_accuracies.append(100 * test_correct / test_total)

    epoch_end_time = time.time()  # Τελος χρονομέτρησης εποχής
    epoch_time = epoch_end_time - epoch_start_time #Χρόνος εποχής
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Acc: {test_accuracies[-1]:.2f}%, Time: {epoch_time:.2f} sec')
end_time = time.time()#Τέλος συνολικής χρονομέτρησης
time = end_time - start_time #Συνολικός Χρόνος
print(f'Total Training Time: {time:.2f} sec')
# Διαγράμματα όπως και στην περίπτωση του συνελικτικού
plt.figure(figsize=(8, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

y_test = []
y_test_pred = []

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_test.extend(labels.cpu().numpy())
        y_test_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testset.classes)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()


def denormalize(image):
    image = image * 0.5 + 0.5
    return np.transpose(image.cpu().numpy(), (1, 2, 0))


correct_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_test_pred)) if true == pred]
incorrect_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_test_pred)) if true != pred]


fig1, axs1 = plt.subplots(2, 2, figsize=(10, 6))
fig1.suptitle("Correct Classifications")
for i in range(4):
    idx = random.choice(correct_indices)
    img, label = testset[idx]
    axs1[i // 2, i % 2].imshow(denormalize(img))
    axs1[i // 2, i % 2].set_title(f"Class: {testset.classes[label]}")
    axs1[i // 2, i % 2].axis('off')

fig2, axs2 = plt.subplots(2, 2, figsize=(10, 6))
fig2.suptitle("Incorrect Classifications")
for i in range(4):
    idx = random.choice(incorrect_indices)
    img, label = testset[idx]
    pred_label = y_test_pred[idx]
    axs2[i // 2, i % 2].imshow(denormalize(img))
    axs2[i // 2, i % 2].set_title(f"True: {testset.classes[label]}, Pred: {testset.classes[pred_label]}")
    axs2[i // 2, i % 2].axis('off')

plt.tight_layout()
plt.show()
