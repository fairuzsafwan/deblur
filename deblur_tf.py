import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import cv2
import numpy as np
from PIL import Image
import time
import argparse
import matplotlib.pyplot as plt
import model as mdl


# Define a custom dataset class
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, image_path_folder, batch_size=32, input_shape=(256, 256), shuffle=True):
        self.image_root_path = image_path_folder
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.image_gt = os.listdir(os.path.join(image_path_folder, "sharp_v3"))
        self.image_train = os.listdir(os.path.join(image_path_folder, "motion_blurred_v3"))

    def __len__(self):
        return int(np.ceil(len(self.image_train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_train = self.image_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_image_gt = self.image_gt[idx * self.batch_size:(idx + 1) * self.batch_size]

        train_images = []
        gt_images = []

        for img_train, img_gt in zip(batch_image_train, batch_image_gt):
            train_image_path = os.path.join(self.image_root_path, "motion_blurred_v3", img_train)
            gt_image_path = os.path.join(self.image_root_path, "sharp_v3", img_gt)

            train_image = read_image(train_image_path, img_size=self.input_shape)
            gt_image = read_image(gt_image_path, img_size=self.input_shape)

            # cv2.imshow("Train", train_image)
            # cv2.imshow("GT", gt_image)
            # cv2.waitKey(0)

            train_images.append(train_image)
            gt_images.append(gt_image)

        return np.array(train_images), np.array(gt_images)

def resizeImage(img, target_size):
    h, w = img.shape[:2]
    aspect_ratio = w/h

    if w > h:
        new_w = int(target_size * aspect_ratio)
        new_h = target_size
    else:
        new_h = int(target_size / aspect_ratio)
        new_w = target_size
    
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #crop image to target_size
    half_cropSize = int(target_size/2)

    h_, w_ = resized_image.shape[:2]
    coor_h_ = int(h_ / 2)
    coor_w_ = int(w_ / 2)

    resized_image = resized_image[coor_h_-half_cropSize:coor_h_+half_cropSize, coor_w_-half_cropSize:coor_w_+half_cropSize]

    return resized_image

def read_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #img = cv2.resize(img, img_size)  # Resize images to a fixed size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img / 255.0  # Normalize the images
    return img

def read_inference_image(image_path, img_size=(256, 256), normalize = False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    
    if h != img_size[0] or w != img_size[1]:
        img = resizeImage(img, img_size[0])  # Resize images to match training input size
    
    if normalize:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype(np.float32) / 255.0  # Normalize the images
    return img

def processDataset(dataset_path, batch_size, input_shape):
    train_dataset = CustomDataset(image_path_folder=dataset_path, batch_size=batch_size, input_shape=input_shape)
    return train_dataset

def trainModel(num_epochs=10, learning_rate=0.001, train_loader=None, model_path="./"):
    print("------------- Training Start -------------")
    best_loss = 10000
    best_epoch = 0
    loss_ls = []

    model = mdl.UNet(n_channels=3, n_classes=3)
    # model = mdl.FPN(n_channels=3, n_classes=3)
    # model = mdl.ResNetUNet(n_channels=3, n_classes=3)
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

        loss_ls.append(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch+1

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')

        #model.save_weights(model_path_name)
        model.save(os.path.join(model_path, str(epoch+1)))

        #Clears sessions for memory leak issues
        tf.keras.backend.clear_session()
    
    #Plot graph
    xs = [x for x in range(len(loss_ls))]
    plt.plot(xs, loss_ls)
    plt.savefig(os.path.join(model_path, "loss.png"))
    plt.close()

    f = open(os.path.join(model_path, "best_epoch.txt"), "w")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Loss: {best_loss}\n")
    f.close()

    end_t = time.time() - start_t
    print(f"Best Model: Epoch {best_epoch} | loss: {best_loss}")
    print(f"Time: {int(end_t // 60)} mins {int(end_t % 60)} secs")

    print("------------- Training completed -------------")

def convertModel(model_path):
    print("------------- Conversion start -------------")
    # Load the model
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(os.path.join(model_path, os.path.basename(model_path)) + ".tflite", "wb") as f:
        f.write(tflite_model)
    print("------------- Conversion completed -------------")

def inference(model_path, img_path, output_path, resized_original_imgpath, img_size):
    print("------------- Inference start -------------")
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    img_paths = [os.path.join(img_path, path) for path in os.listdir(img_path)]

    dirName_path = os.path.dirname(model_path)
    baseName_path = os.path.basename(model_path)
    dirName_path = dirName_path + "_" + baseName_path
    output_path = os.path.join(output_path, dirName_path)
    img_paths = [os.path.join(img_path, path) for path in os.listdir(img_path)]

    if not os.path.exists(output_path):
            os.makedirs(output_path)
        
    if not os.path.exists(resized_original_imgpath):
            os.makedirs(resized_original_imgpath)

    for img_path in img_paths:
        ori_img = read_inference_image(img_path, img_size=img_size, normalize=False)

        img_infer = read_inference_image(img_path, img_size=img_size, normalize=True)
        infer_input = np.expand_dims(img_infer, axis=0)  # Add batch dimension
        output_image = model.predict(infer_input)
        
        # Denormalize the output image
        output_image = output_image.squeeze() * 255.0  

        # Clip and convert image to uint8
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        
        output_image_path = os.path.join(output_path, os.path.basename(img_path))

        resized_ori_imgPath = os.path.join(resized_original_imgpath, os.path.basename(img_path))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if not os.path.exists(resized_original_imgpath):
            os.makedirs(resized_original_imgpath)
        
        

        cv2.imwrite(output_image_path, output_image)
        cv2.imwrite(resized_ori_imgPath, ori_img)
        print(f"Output image saved at {output_image_path}")
        
    print("------------- Inference completed -------------")

if __name__ == "__main__":
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(description="Train deblurring model | Convert model to Tensorflow Lite")
    parser.add_argument("--train", "-t", action="store_true", help="train deblurring model")
    parser.add_argument("--infer", "-i", action="store_true", help="inference")
    parser.add_argument("--convert", "-c", action="store_true", help="convert pytorch model to ONNX and then to Tensorflow Lite")
    parser.add_argument("--path", "-p", default=False, help="path to store saved models")
    args = parser.parse_args()

    # parameters
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 18 #32
    dataset_path = "blur_dataset"
    model_path = "saved_model"
    output_path = "result"
    inference_path = "test_image"
    resized_original_imgpath = "resized_test_image"
    train_loader = None
    img_size = (256, 256) #(256, 256) #(512, 512) (224,224)
    
    #Create directory to save model
    if args.path:
        model_path = args.path
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    print("Model Path: ", model_path)

    # train model
    if args.train:     
        #process dataset
        train_loader = processDataset(dataset_path, batch_size, img_size)

        # Train model
        trainModel(num_epochs, learning_rate, train_loader, model_path)

    if args.convert:
        convertModel(model_path)

    if args.infer:
        inference(model_path, inference_path, output_path, resized_original_imgpath, img_size)
