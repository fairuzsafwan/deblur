import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import cv2
import numpy as np
from PIL import Image
import time
import argparse

# Define a custom dataset class
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, image_path_folder, batch_size=32, input_shape=(256, 256), shuffle=True):
        self.image_root_path = image_path_folder
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.image_gt = os.listdir(os.path.join(image_path_folder, "sharp"))
        self.image_train = os.listdir(os.path.join(image_path_folder, "motion_blurred"))

    def __len__(self):
        return int(np.ceil(len(self.image_train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_train = self.image_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_image_gt = self.image_gt[idx * self.batch_size:(idx + 1) * self.batch_size]

        train_images = []
        gt_images = []

        for img_train, img_gt in zip(batch_image_train, batch_image_gt):
            train_image_path = os.path.join(self.image_root_path, "motion_blurred", img_train)
            gt_image_path = os.path.join(self.image_root_path, "sharp", img_gt)

            train_image = read_image(train_image_path)
            gt_image = read_image(gt_image_path)

            # cv2.imshow("Train", train_image)
            # cv2.imshow("GT", gt_image)
            # cv2.waitKey(0)

            train_images.append(train_image)
            gt_images.append(gt_image)

        return np.array(train_images), np.array(gt_images)

class DoubleConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            layers.ReLU()
        ])

    def call(self, x):
        return self.double_conv(x)

class Down(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2)),
            DoubleConv(in_channels, out_channels)
        ])

    def call(self, x):
        return self.maxpool_conv(x)

class Up(tf.keras.Model):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if not bilinear:
            self.up = layers.Conv2DTranspose(in_channels // 2, kernel_size=2, strides=2, padding='same')
        self.conv = DoubleConv(in_channels, out_channels)

    def call(self, x1, x2):
        if self.bilinear:
            x1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
        else:
            x1 = self.up(x1)
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]
        x1 = layers.ZeroPadding2D(padding=((diffX // 2, diffX - diffX // 2), (diffY // 2, diffY - diffY // 2)))(x1)
        x = tf.concat([x2, x1], axis=-1)
        return self.conv(x)

class UNet(tf.keras.Model):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = layers.Conv2D(n_classes, kernel_size=1)

    def call(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        results = self.outc(x)
        return results

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize images to a fixed size
    img = img / 255.0  # Normalize the images
    return img

def read_inference_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))  # Resize images to match training input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0  # Normalize the images
    return img

def processDataset(dataset_path, batch_size):
    train_dataset = CustomDataset(image_path_folder=dataset_path, batch_size=batch_size)
    return train_dataset

def trainModel(num_epochs=10, learning_rate=0.001, train_loader=None, model_path="./"):
    print("------------- Training Start -------------")
    model = UNet(n_channels=3, n_classes=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criterion = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(train_img, gt_img):
        with tf.GradientTape() as tape:
            pred_img = model(train_img)
            loss = criterion(gt_img, pred_img)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    start_t = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_images, batch_gt_images in train_loader:
            loss = train_step(batch_images, batch_gt_images)
            total_loss += loss

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')
    
    end_t = time.time() - start_t
    print(f"Time: {int(end_t // 60)} mins {int(end_t % 60)} secs")

    #model.save_weights(model_path_name)
    model.save(model_path)
    print("------------- Training completed -------------")

def convertModel(model_path):
    print("------------- Conversion start -------------")
    # Load the model
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(os.path.join(model_path, model_path) + ".tflite", "wb") as f:
        f.write(tflite_model)
    print("------------- Conversion completed -------------")

def inference(model_path, img_path, output_path):
    print("------------- Inference start -------------")
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    img_infer = read_inference_image(img_path)
    infer_input = np.expand_dims(img_infer, axis=0)  # Add batch dimension
    output_image = model.predict(infer_input)
    
    # Denormalize the output image
    output_image = output_image.squeeze() * 255.0  

    # Clip and convert image to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image_path = os.path.join(output_path, os.path.basename(img_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_image_path, output_image)
    print(f"Output image saved at {output_image_path}")
    print("------------- Inference completed -------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deblurring model | Convert model to Tensorflow Lite")
    parser.add_argument("--train", "-t", action="store_true", help="train deblurring model")
    parser.add_argument("--infer", "-i", action="store_true", help="inference")
    parser.add_argument("--convert", "-c", action="store_true", help="convert pytorch model to ONNX and then to Tensorflow Lite")
    parser.add_argument("--path", "-p", default=False, help="path to store saved models")
    args = parser.parse_args()

    # parameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 24 #32
    dataset_path = "blur_dataset"
    model_path = "saved_model2"
    output_path = "result"
    inference_path = "test_image/6_HUAWEI-MATE20_M.JPG"
    train_loader = None
    
    #Create directory to save model
    if args.path:
        model_path = args.path
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("Model Path: ", model_path)

    # train model
    if args.train:     
        #process dataset
        train_loader = processDataset(dataset_path, batch_size)

        # Train model
        trainModel(num_epochs, learning_rate, train_loader, model_path)

    if args.convert:
        convertModel(model_path)

    if args.infer:
        inference(model_path, inference_path, output_path)