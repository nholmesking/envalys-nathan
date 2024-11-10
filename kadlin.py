import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import jarnbra
import landbjartur

"""
Command-line arguments:
1. Directory for images
2. Epochs for first model
3. Epochs for second model

PEP-8 compliant.
"""

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
no_points_embedding = nn.Parameter(torch.randn(128).to(device))


# Define a CNN for image processing (ResNet18 as an example)
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Using a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the last layer to output a 128-dimensional feature vector
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, images):
        return self.resnet(images)


# Define the LSTM-based embedding model for coordinates
class CoordinateEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CoordinateEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, coordinates):
        lstm_out, _ = self.lstm(coordinates)
        embedding = self.fc(lstm_out[:, -1, :])  # Last hidden state
        return embedding


# Combine both models into a final training model
class SatelliteModel(nn.Module):
    def __init__(self, image_encoder, coordinate_encoder, embedding_size=128):
        super(SatelliteModel, self).__init__()
        self.image_encoder = image_encoder
        self.coordinate_encoder = coordinate_encoder
        # Define a special embedding for "no points" (either zero or learnable)
        self.no_points_embedding = no_points_embedding

    def forward(self, images, coordinates=None):
        # Encode the image
        image_features = self.image_encoder(images)  # Shape: (batch_size, 128)

        if coordinates is not None and coordinates.shape[1] > 0:
            # If coordinates exist, get the coordinate embedding
            coordinate_embedding = self.coordinate_encoder(coordinates)
        else:
            # If no coordinates, use the "no points" embedding
            coordinate_embedding = self.no_points_embedding.\
                                   expand(images.size(0), -1)

        return image_features, coordinate_embedding


def train_SatelliteModel(X_train, X_test, y_train, y_test, epochs):
    byrjun = time.time()
    # Initialize both models
    image_encoder = ImageEncoder()
    coordinate_encoder = CoordinateEmbedding(input_size=2, hidden_size=128,
                                             output_size=128)

    # Combine into final model
    likan = SatelliteModel(image_encoder, coordinate_encoder).to(device)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(likan.parameters(), lr=1e-4)

    train_loss = []
    test_loss = []

    for e in range(epochs):
        likan.eval()
        with torch.no_grad():
            rl = 0.0
            for i in range(len(X_train)):
                image_features, coordinate_embedding = likan(X_train[i],
                                                             y_train[i])
                loss = criterion(image_features, coordinate_embedding)
                rl += loss.item()
            train_loss.append(rl / len(X_train))
            rl = 0.0
            for i in range(len(X_test)):
                image_features, coordinate_embedding = likan(X_test[i],
                                                             y_test[i])
                loss = criterion(image_features, coordinate_embedding)
                rl += loss.item()
            test_loss.append(rl / len(X_test))
        torch.mps.empty_cache()

        likan.train()
        for i in range(len(X_train)):
            # Forward pass: get image features and coordinate embeddings
            image_features, coordinate_embedding = likan(X_train[i],
                                                         y_train[i])

            # Loss: comparing image features with coordinate embeddings
            loss = criterion(image_features, coordinate_embedding)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if i % 10 == 0:
                torch.mps.empty_cache()

        torch.mps.empty_cache()

        # Print loss for the current epoch
        print(f'Epoch [{e+1}/{epochs}], Time: {time.time() - byrjun}')

    likan.eval()
    with torch.no_grad():
        rl = 0.0
        for i in range(len(X_train)):
            image_features, coordinate_embedding = likan(X_train[i],
                                                         y_train[i])
            loss = criterion(image_features, coordinate_embedding)
            rl += loss.item()
        train_loss.append(rl / len(X_train))
        rl = 0.0
        for i in range(len(X_test)):
            image_features, coordinate_embedding = likan(X_test[i], y_test[i])
            loss = criterion(image_features, coordinate_embedding)
            rl += loss.item()
        test_loss.append(rl / len(X_test))
    torch.mps.empty_cache()

    print('Train loss:', train_loss[-1])
    print('Test loss:', test_loss[-1])

    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.ylim(bottom=0)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_loss[1:], label='Train')
    plt.plot(test_loss[1:], label='Test')
    plt.ylim(bottom=0)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def chamfer_distance(pred_points, target_points):
    """
    pred_points: List of tensors of shape (Ni, D), where Ni is the number of
        points in the i-th prediction.
    target_points: List of tensors of shape (Mi, D), where Mi is the number of
        points in the i-th target.
    """
    total_loss = 0.0
    batch_size = len(pred_points)
    for pred, target in zip(pred_points, target_points):
        if pred.numel() == 0 and target.numel() == 0:
            # Both sequences are empty; loss is zero.
            loss = 0.0
        elif pred.numel() == 0 or target.numel() == 0:
            # One sequence is empty
            non_empty = pred if pred.numel() != 0 else target
            loss = non_empty.pow(2).sum()
        else:
            # Compute pairwise distances.
            diff = pred.unsqueeze(1) - target.unsqueeze(0)  # (Ni, Mi, D)
            dist = torch.norm(diff, dim=2)  # Shape: (Ni, Mi)
            # For each point in pred, find the closest point in target.
            min_pred_to_target = dist.min(dim=1)[0]
            # For each point in target, find the closest point in pred.
            min_target_to_pred = dist.min(dim=0)[0]
            # Sum the minimal distances.
            loss = min_pred_to_target.sum() + min_target_to_pred.sum()
        total_loss += loss
    return total_loss / batch_size


def pointToLineDist(point, line):
    """
    Calculate the shortest distance between a point and a line segment in 2D
    space. If the perpendicular projection of the point onto the line does not
    fall within the line segment, the distance to the nearest endpoint is
    returned.

    Parameters:
    - point: numpy array of shape (2,) for coordinates of point.
    - line: numpy array of shape (4,) for coordinates of line segment
            in the form [x1, y1, x2, y2].

    Returns:
    - distance: The shortest distance between the point and the line segment.
    """
    # Extract points
    x1, y1, x2, y2 = line
    A = np.array([x1, y1])
    B = np.array([x2, y2])
    P = point

    # Compute vectors
    AB = B - A
    AP = P - A
    norm_AB_squared = np.dot(AB, AB)

    # Handle the case where A and B are the same point
    if norm_AB_squared == 0:
        return np.linalg.norm(P - A)

    # Compute the projection parameter t
    t = np.dot(AP, AB) / norm_AB_squared

    # Determine the closest point on the line segment to P
    if t < 0.0:
        closest_point = A
    elif t > 1.0:
        closest_point = B
    else:
        closest_point = A + t * AB

    # Compute and return the distance from P to the closest point
    distance = np.linalg.norm(P - closest_point)
    return distance


def chamfer_vegir(pred_lines, target_lines):
    if len(pred_lines) == 0 and len(target_lines) == 0:
        return 0.0
    total_loss = 0.0
    # For each pred point: distance to closest target line
    for i in range(pred_lines.shape[0]):
        minDist = 2 ** target_lines.shape[0]
        for j in range(target_lines.shape[0]):
            if abs(pred_lines[i][0] - target_lines[j][0]) > minDist and \
               abs(pred_lines[i][0] - target_lines[j][2]) > minDist:
                continue
            if abs(pred_lines[i][1] - target_lines[j][1]) > minDist and \
               abs(pred_lines[i][1] - target_lines[j][3]) > minDist:
                continue
            d = pointToLineDist(pred_lines[i][0:2], target_lines[j])
            if d < minDist:
                minDist = d
        total_loss += minDist
    # For each target point: distance to closest pred line
    for i in range(target_lines.shape[0]):
        minDist = 2 ** target_lines.shape[0]
        for j in range(pred_lines.shape[0]):
            if abs(target_lines[i][0] - pred_lines[j][0]) > minDist and \
               abs(target_lines[i][0] - pred_lines[j][2]) > minDist:
                continue
            if abs(target_lines[i][1] - pred_lines[j][1]) > minDist and \
               abs(target_lines[i][1] - pred_lines[j][3]) > minDist:
                continue
            d = pointToLineDist(target_lines[i][0:2], pred_lines[j])
            if d < minDist:
                minDist = d
        total_loss += minDist
    # Penalty for pred line length
    for j in range(pred_lines.shape[0]):
        total_loss += (pred_lines[j][0] - pred_lines[j][2]) ** 2 + \
                      (pred_lines[j][1] - pred_lines[j][3]) ** 2
    return total_loss


class CoordinateDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size=2,
                 num_layers=1):
        super(CoordinateDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The initial hidden state is derived from the embedding
        self.fc_embed = nn.Linear(embedding_size, hidden_size * num_layers)

        # LSTM decoder
        self.lstm = nn.LSTM(input_size=output_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_stop = nn.Linear(hidden_size, 1)  # Pred. stop probability

    def forward(self, embedding, max_seq_length):
        batch_size = embedding.size(0)

        # Initialize hidden state from embedding
        h_0 = self.fc_embed(embedding)
        h_0 = h_0.view(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros_like(h_0)

        # Prepare the initial input (e.g., start token or zeros)
        decoder_input = torch.zeros(batch_size, 1, 2).to(embedding.device)

        outputs = []
        stop_probs = []
        for t in range(max_seq_length):
            # LSTM step
            output, (h_0, c_0) = self.lstm(decoder_input, (h_0, c_0))
            coord_pred = self.fc_out(output)
            stop_logit = self.fc_stop(output)
            stop_prob = torch.sigmoid(stop_logit)

            outputs.append(coord_pred)
            stop_probs.append(stop_prob)

            # Check if all sequences have predicted stop
            if torch.all(stop_prob > 0.5):
                break

            decoder_input = coord_pred  # Or use teacher forcing

        if len(outputs) == 0:
            outputs = torch.tensor(np.zeros((1, 0, 2)),
                                   dtype=torch.float32).to(device)
            stop_probs = torch.tensor(np.zeros((0, 1)),
                                      dtype=torch.float32).to(device)
            return outputs, stop_probs

        outputs = torch.cat(outputs, dim=1)
        stop_probs = torch.cat(stop_probs, dim=1)
        return outputs, stop_probs


def train_CoordinateDecoder(X_train, X_test, y_train, y_test, likan, epochs):
    byrjun = time.time()
    # Initialize the decoder
    coordinate_decoder = CoordinateDecoder(embedding_size=128,
                                           hidden_size=128).to(device)

    # Define optimizer
    decoder_optimizer = optim.Adam(coordinate_decoder.parameters(), lr=1e-4)

    train_loss = []
    test_loss = []

    # Training loop for the decoder
    likan.eval()
    for e in range(epochs):
        coordinate_decoder.eval()
        with torch.no_grad():
            rl = 0.0
            for i in range(len(y_train)):
                coord_embeddings = likan.coordinate_encoder(y_train[i])
                seq_length = y_train[i].size(1)
                output_sequences, stop_probs = \
                    coordinate_decoder(coord_embeddings, seq_length)
                loss = chamfer_distance(output_sequences, y_train[i])
                rl += loss.item()
            train_loss.append(rl / len(y_train))
            rl = 0.0
            for i in range(len(y_test)):
                coord_embeddings = likan.coordinate_encoder(y_test[i])
                seq_length = y_train[i].size(1)
                output_sequences, stop_probs = \
                    coordinate_decoder(coord_embeddings, seq_length)
                loss = chamfer_distance(output_sequences, y_test[i])
                rl += loss.item()
            test_loss.append(rl / len(y_test))

        torch.mps.empty_cache()

        coordinate_decoder.train()
        for i in range(len(y_train)):
            # Get image embeddings
            with torch.no_grad():
                coord_embeddings = likan.coordinate_encoder(y_train[i])

            # Forward pass thru decoder
            seq_length = y_train[i].size(1)
            output_sequences, stop_probs = coordinate_decoder(coord_embeddings,
                                                              seq_length)

            # Compute loss
            loss = chamfer_distance(output_sequences, y_train[i])

            # Backpropagation and optimization
            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            decoder_optimizer.step()
            if i % 10 == 0:
                torch.mps.empty_cache()

        gc.collect()
        torch.mps.empty_cache()

        print(f'Epoch [{e+1}/{epochs}], Time: {time.time() - byrjun}')

    coordinate_decoder.eval()
    with torch.no_grad():
        rl = 0.0
        for i in range(len(y_train)):
            coord_embeddings = likan.coordinate_encoder(y_train[i])
            seq_length = y_train[i].size(1)
            output_sequences, stop_probs = coordinate_decoder(coord_embeddings,
                                                              seq_length)
            loss = chamfer_distance(output_sequences, y_train[i])
            rl += loss.item()
        train_loss.append(rl / len(y_train))
        rl = 0.0
        for i in range(len(y_test)):
            coord_embeddings = likan.coordinate_encoder(y_test[i])
            seq_length = y_train[i].size(1)
            output_sequences, stop_probs = coordinate_decoder(coord_embeddings,
                                                              seq_length)
            loss = chamfer_distance(output_sequences, y_test[i])
            rl += loss.item()
        test_loss.append(rl / len(y_test))
        torch.mps.empty_cache()

    print('Train loss:', train_loss[-1])
    print('Test loss:', test_loss[-1])

    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.ylim(bottom=0)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    hnitlisti = jarnbra.getCoordList(sys.argv[1])
    X_gogn, y_gogn = landbjartur.getRoadData('U', sys.argv[1], hnitlisti)
    X_train, X_test, y_train, y_test = train_test_split(X_gogn, y_gogn)
    sm_likan = train_SatelliteModel(X_train, X_test, y_train, y_test,
                                    int(sys.argv[2]))
    for i in range(len(X_gogn)):
        X_gogn[i] = None
    del X_gogn
    for i in range(len(X_train)):
        X_train[i] = None
    del X_train
    for i in range(len(X_test)):
        X_test[i] = None
    del X_test
    gc.collect()
    torch.mps.empty_cache()
    train_CoordinateDecoder(X_train, X_test, y_train, y_test, sm_likan,
                            int(sys.argv[3]))
