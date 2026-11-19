from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import tempfile
import torchaudio
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir, max_samples=None):
        self.data = pd.read_csv(csv_path)
        if max_samples:
            self.data = self.data.iloc[:max_samples].reset_index(drop=True)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "surprise": 5,
            "fear": 6
        }

        # positive, negative and neutral
        self.sentiment_map = {
            "positive": 2,
            "negative": 0,
            "neutral": 1
        }

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")
            
            # Try to read the first frame to ensure the video is valid
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Normalize to [0, 1]
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error video :  {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Pad or truncate to frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))  # Pad with zeros
        else:
            frames = frames[:30]  # Truncate to 30 frames

        # Before permute : [30, 224, 224, 3] (frames, height, width, channels)
        # After permute : [30, 3, 224, 224] (frames, channels, height, width)
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)  # Convert to tensor and change shape to (frames, channels, height, width)
    
    def _extract_audio_features(self, video_path):
        # Temp file (not next to the video) so dataset dirs stay clean/read-only-safe
        # and parallel workers never collide.
        fd, audio_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

        try:
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)   # shape (channels, frames), same as torchaudio.load

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            #waveform, sample_rate = torchaudio.load(audio_path)  # Load the audio to ensure it's valid

            #if sample_rate != 16000:
            #    resampled_waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            #    waveform = resampled_waveform

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
                )(waveform)

            # Normalize the Mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec  # Shape: [1, 64, 300]
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio Error : {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)  # Clean up the temporary audio file
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Convert tensor to int if necessary
            
        row = self.data.iloc[idx]

        try :
            video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(path)
            if not video_path_exists:
                raise FileNotFoundError(f"No video found for filename: {path}")

            text_inputs = self.tokenizer(row['Utterance'], return_tensors="pt", padding='max_length', truncation=True, max_length=128)

            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            # Map emotions and sentiments to their respective indices
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    "input_ids": text_inputs['input_ids'].squeeze(),  # Remove batch dimension
                    "attention_mask": text_inputs['attention_mask'].squeeze(),  # Remove batch dimension
                },
                "video_frames": video_frames,  # Shape: [30, 3, 224, 224]
                "audio_features": audio_features,  # Shape: [1, 64, 300]
                "emotion_label": torch.tensor(emotion_label),
                "sentiment_label": torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {path} : {str(e)}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_dataloaders(train_csv,train_video_dir,
                        dev_csv,dev_video_dir,
                        test_csv,test_video_dir, batch_size=32,
                        max_samples=None, num_workers=0):

    train_dataset = MELDDataset(train_csv,train_video_dir, max_samples)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir, max_samples)
    test_dataset = MELDDataset(test_csv, test_video_dir, max_samples)

    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate_fn,
                         num_workers=num_workers,
                         persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        "../dataset/train/train_sent_emo.csv", "../dataset/train/train_splits/",
        "../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete/",
        "../dataset/test/test_sent_emo.csv", "../dataset/test/output_repeated_splits_test/"
    )

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break