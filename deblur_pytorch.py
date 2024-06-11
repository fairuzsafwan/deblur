import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import argparse

import model_py as mdl

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_path_folder, input_shape=(256, 256)):
        self.image_root_path = image_path_folder
        self.input_shape = input_shape
        self.image_gt = os.listdir(os.path.join(image_path_folder, "sharp"))
        self.image_train = os.listdir(os.path.join(image_path_folder, "motion_blurred_v2"))

    def __len__(self):
        return len(self.image_train)

    def __getitem__(self, idx):
        img_train = self.image_train[idx]
        img_gt = self.image_gt[idx]

        train_image_path = os.path.join(self.image_root_path, "motion_blurred_v2", img_train)
        gt_image_path = os.path.join(self.image_root_path, "sharp", img_gt)

        train_image = read_image(train_image_path, img_size=self.input_shape)
        gt_image = read_image(gt_image_path, img_size=self.input_shape)

        return train_image, gt_image

def read_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)  # Resize images to a fixed size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img / 255.0  # Normalize the images
    img = np.transpose(img, (2, 0, 1))  # Change to CHW format
    return torch.tensor(img, dtype=torch.float32)



def trainModel(num_epochs=10, learning_rate=0.001, train_loader=None, model_path="./"):
    print("------------- Training Start -------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = mdl.UNet(n_channels=3, n_classes=3).to(device)
    model = mdl.FPN(n_channels=3, n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_t = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for batch_images, batch_gt_images in train_loader:
            batch_images = batch_images.to(device)
            batch_gt_images = batch_gt_images.to(device)

            optimizer.zero_grad()
            pred_img = model(batch_images)
            loss = criterion(pred_img, batch_gt_images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')

        torch.save(model.state_dict(), os.path.join(model_path, f"model_epoch_{epoch+1}.pth"))
    
    end_t = time.time() - start_t
    print(f"Time: {int(end_t // 60)} mins {int(end_t % 60)} secs")

    print("------------- Training completed -------------")

def inference(model_path, img_path, output_path, resized_original_imgpath, img_size):
    print("------------- Inference start -------------")

    model_name = os.path.dirname(model_path)
    model_epoch = os.path.basename(model_path).split(".")[0]
    model_path_name = model_name + "_" + model_epoch
    output_path = os.path.join(output_path, model_path_name)

    if not os.path.exists(output_path):
            os.makedirs(output_path)
        
    if not os.path.exists(resized_original_imgpath):
        os.makedirs(resized_original_imgpath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mdl.UNet(n_channels=3, n_classes=3).to(device)
    model = mdl.FPN(n_channels=3, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img_paths = [os.path.join(img_path, path) for path in os.listdir(img_path)]

    for img_path in img_paths:
        ori_img = read_inference_image(img_path, img_size=img_size, normalize=False)
        ori_img = ori_img.cpu().detach().numpy()
        img_infer = read_inference_image(img_path, img_size=img_size, normalize=True)
        infer_input = torch.unsqueeze(img_infer, 0).to(device)

        with torch.no_grad():
            output_image = model(infer_input).squeeze(0).cpu().numpy()
        
        output_image = np.transpose(output_image, (1, 2, 0)) * 255.0  # CHW to HWC and denormalize
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        output_image_path = os.path.join(output_path, os.path.basename(img_path))
        resized_ori_imgPath = os.path.join(resized_original_imgpath, os.path.basename(img_path))

        

        cv2.imwrite(output_image_path, output_image)
        cv2.imwrite(resized_ori_imgPath, ori_img)
        print(f"Output image saved at {output_image_path}")

    print("------------- Inference completed -------------")

def read_inference_image(image_path, img_size=(256, 256), normalize=False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    if normalize:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return torch.tensor(img, dtype=torch.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deblurring model | Convert model to ONNX and then to Tensorflow Lite")
    parser.add_argument("--train", "-t", action="store_true", help="train deblurring model")
    parser.add_argument("--infer", "-i", action="store_true", help="inference")
    parser.add_argument("--convert", "-c", action="store_true", help="convert pytorch model to ONNX and then to Tensorflow Lite")
    parser.add_argument("--path", "-p", default=False, help="path to store saved models")
    args = parser.parse_args()

    # parameters
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 18
    dataset_path = "blur_dataset"
    model_path = "saved_model"
    output_path = "result"
    inference_path = "test_image"
    resized_original_imgpath = "resized_test_image"
    img_size = (256, 256)

    if args.path:
        model_path = args.path
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("Model Path: ", model_path)

    if args.train:
        train_dataset = CustomDataset(image_path_folder=dataset_path, input_shape=img_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        trainModel(num_epochs, learning_rate, train_loader, model_path)

    if args.infer:
        inference(model_path, inference_path, output_path, resized_original_imgpath, img_size)
