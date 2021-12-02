import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ImageClassificationCollator:
  def __init__(self, feature_extractor):
      self.feature_extractor = feature_extractor

  def __call__(self, batch):
      encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
      encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
      return encodings 


class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy()
        self.train_acc= Accuracy()

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        acc1 = self.train_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"train_acc", acc1, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,weight_decay=0.0025)


class Model:
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        self.collator = ImageClassificationCollator(self.feature_extractor)
        self.classifier = Classifier(self.model, lr=2e-5)
        self.trainer = pl.Trainer(max_epochs=5)
        
    def predict(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]

    def train(self, df):
        print("[*] Model.train(df)")
        print(df)
        # Preprocessing
        df["images"] = df["image_url"].apply(lambda image_url: Image.open(requests.get(image_url, stream=True).raw))
        df["encodings"] = df["images"].apply(lambda image: self.feature_extractor(images=image, return_tensors="pt"))
        df["labels"] = df["label"].apply(lambda label: self.model.config.label2id[label])
        # Create datasets
        train_dataset, eval_dataset = torch.utils.data.random_split(df[["images", "labels"]].values.tolist(), [9, 1])
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=self.collator, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, collate_fn=self.collator)
        # Train
        pl.seed_everything(42)
        self.trainer.fit(self.classifier, train_loader, eval_loader)
        torch.save(self.model, 'model.pt')
