import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
class HybridModel(nn.Module):
    def __init__(self, wav2vec_model_name='facebook/wav2vec2-base', hidden_size=768, num_classes=2):
        super(HybridModel, self).__init__()
        self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_model_name, num_labels=hidden_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn_output_size = 128 * 16 * 16
        self.cnn_projection = nn.Linear(self.cnn_output_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, spectrogram):
        wav2vec_output = self.wav2vec(inputs).logits
        batch_size = spectrogram.size(0)
        cnn_output = self.cnn(spectrogram)
        cnn_output = cnn_output.view(batch_size, -1)
        cnn_projected = self.cnn_projection(cnn_output)
        combined = torch.cat((wav2vec_output, cnn_projected), dim=1)
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output
