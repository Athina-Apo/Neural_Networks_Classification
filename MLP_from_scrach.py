# Η βιβλιοθήκη torchvision χρησιμοποιήθηκε μόνο για την φόρτωση του data set
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

# Load the entire training set into memory
for data in trainloader:
    X_train, y_train = data
    break
# Load the entire test set into memory
for data in testloader:
    X_test, y_test = data
    break

# Flatten the images and convert to NumPy arrays
X_train = X_train.view(X_train.size(0), -1).numpy()
y_train = y_train.numpy()
X_test = X_test.view(X_test.size(0), -1).numpy()
y_test = y_test.numpy()

# Αντιστοιχίζει τα labels σε έναν αριθμό
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)

#Η συνάρτηση relu
def relu(x):
    return np.maximum(0, x)
#Η παράγωγος της relu
def relu_gradient(x):
    return (x > 0).astype(float)


#H sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#H παράγωγος της sigmoid
def sigmoid_gradient(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

#H συνάρτηση sofmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Ορίζω το νεωρωνικό δίκτυο ως κλάση
class FullyConnectedNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.04):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01 #Ο πίνακας με τα βάρη ανάμεσα στην είσοδο και το κρυφό layer
        self.b1 = np.zeros((1, hidden_size)) # Tα αντίστοιχα bias
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 #Ο πίνακας με τα βάρη ανάμεσα στο κρυφό layer και την έξοδο
        self.b2 = np.zeros((1, output_size)) # Tα αντίστοιχα bias
        self.learning_rate = learning_rate #Το learning rate
        self.train_accuracies = []  # Αποθηκεύω training accuracies
        self.test_accuracies = []   # Αποθηκεύω test accuracies
        self.train_losses = []      # Αποθηκεύω training losses
        self.test_losses = []       # Αποθηκεύω test losses

    def forward(self, X):

        #Εμπρόσθιο πέρασμα της εισόδου στο νευρωνικό
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1) #Για ενδιάμεση συνάρτηση ενεργοποίησης είτε relu είτε sigmoid
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        m = X.shape[0]
        y_pred = self.A2

        # Υπολογισμός error στο εξωτερικό layer
        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Υπολογισμός error στο hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_gradient(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Ανανέωση βαρών και biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def compute_loss(self, y_true, y_pred):#Συνάρτηση απώλειας cross entropy
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))  #Προσθέτω στο τέλος μια μικρή σταθερά

    #Συνάρτηση εκπαίδευσης
    def train(self, X, y, epochs=100, decay_factor=0.95):
        start_time = time.time() #Ξεκινάει ο μετρητής του χρόνου εκπαίδευσης
        for epoch in range(epochs):
            if epoch % 20 == 0:
                self.learning_rate *= decay_factor #Κάθε 20 εποχές μειώνεται το learning rate
                # Όταν θέλω σταθερό learning rate βαζω decay_factor=1
            # Forward pass
            y_pred = self.forward(X)
            # Υπολογισμός training loss
            train_loss = self.compute_loss(y, y_pred)
            self.train_losses.append(train_loss)

            # Backward pass
            self.backward(X, y)

            # Υπολογισμός training accuracy
            train_accuracy = self.accuracy(X, np.argmax(y, axis=1))
            self.train_accuracies.append(train_accuracy)

            # Υπολογιμός test loss και accuracy
            y_test_pred = self.forward(X_test)  # Reuse forward pass for the test set
            test_loss = self.compute_loss(y_test_one_hot, y_test_pred)
            self.test_losses.append(test_loss)
            test_accuracy = self.accuracy(X_test, y_test)
            self.test_accuracies.append(test_accuracy)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
            end_time = time.time()#Τέλος χρονομέτρησης
        self.training_time = end_time - start_time  # Χρόνος εκπαίδευσης
        print(f'Training completed in {self.training_time:.2f} seconds')

    def predict(self, X):
        # Πρόβλεψη κλάσης
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)  # Επιλέγει την κλάση με τη μεγαλύτερη πιθανότητα

    def accuracy(self, X, y):
        #Υπολογίζει την ακρίβεια
        y_pred = self.predict(X)
        correct = np.sum(y_pred == y)
        return correct / len(y)


input_size = 3072  # 32*32*3 για CIFAR-10
hidden_size = 1024 #Ελεύθερη επιλογή
output_size = 10 #Αφού έχουμε 10 εξόδους

nn = FullyConnectedNN(input_size, hidden_size, output_size)

# Κανονικοποίηση X_train και X_test σε διάστημα [0, 1]
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# Train
nn.train(X_train, y_train_one_hot, epochs=300)

#Αccuracy on the test set
test_accuracy = nn.accuracy(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Τα διάφορα διαγράμματα

# Διάγραμμα accuracy
plt.figure(figsize=(10, 6))
plt.plot(nn.train_accuracies, label='Training Accuracy')
plt.plot(nn.test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


# Διάγραμμα απώλειας
plt.figure(figsize=(10, 6))
plt.plot(nn.train_losses, label='Training Loss')
plt.plot(nn.test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


# Διάγραμμα σύγχυσης
y_test_pred = nn.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trainset.classes)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Σωστά παραδείγματα
correct_indices = np.where(y_test == y_test_pred)[0]
incorrect_indices = np.where(y_test != y_test_pred)[0]

fig1, axs1 = plt.subplots(2, 2, figsize=(10, 6))  # Two rows and two columns for correct examples
fig1.suptitle("Correct Classifications")

for i in range(4):  # 4 σωστά παραδείγματα
    idx_correct = correct_indices[np.random.randint(0, len(correct_indices))]
    img = X_test[idx_correct].reshape(3, 32, 32).transpose(1, 2, 0) * 0.5 + 0.5
    row = i // 2
    col = i % 2
    axs1[row, col].imshow(img)
    axs1[row, col].set_title(f"Correct: {trainset.classes[y_test[idx_correct]]}")
    axs1[row, col].axis('off')

# Τυχαία παραδείγματα - Incorrect Classifications
fig2, axs2 = plt.subplots(2, 2, figsize=(10, 6))
fig2.suptitle("Incorrect Classifications")

for i in range(4):  # 4 λάθος παραδείγματα
    idx_incorrect = incorrect_indices[np.random.randint(0, len(incorrect_indices))]
    img = X_test[idx_incorrect].reshape(3, 32, 32).transpose(1, 2, 0) * 0.5 + 0.5
    row = i // 2
    col = i % 2
    axs2[row, col].imshow(img)
    axs2[row, col].set_title(f"True: {trainset.classes[y_test[idx_incorrect]]}\nPred: {trainset.classes[y_test_pred[idx_incorrect]]}")
    axs2[row, col].axis('off')

# Show both plots
plt.tight_layout()
plt.show()
