import torch
from transformers import Wav2Vec2Model
import torch.nn as nn

class StutterDetector(nn.Module):
    def __init__(self, wav2vec_model_name='facebook/wav2vec2-base'):
        super(StutterDetector, self).__init__()

        # Wav2vec2 for raw audio
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)

        # CNN for spectrogram
        self.cnn = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling for flexible input sizes
            nn.Dropout2d(0.2)
        )

        # Projection layers
        self.wav2vec_projection = nn.Sequential(
            nn.Linear(768, 512),  # 768 is wav2vec2-base hidden size
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.cnn_projection = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 classes: stuttering vs non-stuttering
        )

    def forward(self, audio_input, spectrogram_input):
        # Process raw audio through wav2vec2
        wav2vec_output = self.wav2vec(audio_input).last_hidden_state
        wav2vec_pooled = torch.mean(wav2vec_output, dim=1)  # Average pooling over sequence length
        wav2vec_features = self.wav2vec_projection(wav2vec_pooled)

        # Process spectrogram through CNN
        batch_size = spectrogram_input.size(0)
        cnn_output = self.cnn(spectrogram_input)
        cnn_flattened = cnn_output.view(batch_size, -1)
        cnn_features = self.cnn_projection(cnn_flattened)

        # Combine features and classify
        combined_features = torch.cat((wav2vec_features, cnn_features), dim=1)
        output = self.classifier(combined_features)

        return output
