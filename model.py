import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import ResNet50

#------------------------------------------------FPN ----------------------------------------------------
# class DoubleConv(tf.keras.Model):
#     def __init__(self, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = tf.keras.Sequential([
#             layers.Conv2D(out_channels, kernel_size=3, padding='same'),
#             layers.ReLU(),
#             layers.Conv2D(out_channels, kernel_size=3, padding='same'),
#             layers.ReLU()
#         ])

#     def call(self, x):
#         return self.double_conv(x)

# class Down(tf.keras.Model):
#     def __init__(self, out_channels):
#         super(Down, self).__init__()
#         self.maxpool_conv = tf.keras.Sequential([
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             DoubleConv(out_channels)
#         ])

#     def call(self, x):
#         return self.maxpool_conv(x)

# class FPN(tf.keras.Model):
#     def __init__(self, n_channels, n_classes):
#         super(FPN, self).__init__()
#         self.inc = DoubleConv(64)
#         self.down1 = Down(128)
#         self.down2 = Down(256)
#         self.down3 = Down(512)
#         self.down4 = Down(512)

#         # Lateral layers
#         self.latlayer1 = layers.Conv2D(256, kernel_size=1, padding='same')
#         self.latlayer2 = layers.Conv2D(256, kernel_size=1, padding='same')
#         self.latlayer3 = layers.Conv2D(256, kernel_size=1, padding='same')
#         self.latlayer4 = layers.Conv2D(256, kernel_size=1, padding='same')

#         # Smooth layers
#         self.smooth1 = layers.Conv2D(256, kernel_size=3, padding='same')
#         self.smooth2 = layers.Conv2D(256, kernel_size=3, padding='same')
#         self.smooth3 = layers.Conv2D(256, kernel_size=3, padding='same')

#         self.outc = layers.Conv2D(n_classes, kernel_size=1)

#     def call(self, x):
#         # Bottom-up pathway
#         c1 = self.inc(x)  # 256x256x64
#         c2 = self.down1(c1)  # 128x128x128
#         c3 = self.down2(c2)  # 64x64x256
#         c4 = self.down3(c3)  # 32x32x512
#         c5 = self.down4(c4)  # 16x16x512

#         # Top-down pathway and lateral connections
#         p5 = self.latlayer1(c5)  # 16x16x256
#         p4 = self.latlayer2(c4) + tf.image.resize(p5, size=(tf.shape(c4)[1], tf.shape(c4)[2]))  # 32x32x256
#         p4 = self.smooth1(p4)
#         p3 = self.latlayer3(c3) + tf.image.resize(p4, size=(tf.shape(c3)[1], tf.shape(c3)[2]))  # 64x64x256
#         p3 = self.smooth2(p3)
#         p2 = self.latlayer4(c2) + tf.image.resize(p3, size=(tf.shape(c2)[1], tf.shape(c2)[2]))  # 128x128x256
#         p2 = self.smooth3(p2)

#         p1 = tf.image.resize(p2, size=(tf.shape(c1)[1], tf.shape(c1)[2]))  # 256x256x256

#         results = self.outc(p1)  # 256x256x<n_classes>
#         return results


#------------------------------------------------ UNET with ResNet 50 backbone ----------------------------------------------------
class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv_residual = layers.Conv2D(filters, kernel_size=1, padding='same')  # Match dimensions for addition

    def call(self, inputs):
        residual = self.conv_residual(inputs)  # Ensure dimensions match
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class UpSampleBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, upsample=True):
        super(UpSampleBlock, self).__init__()
        if upsample:
            self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        else:
            self.upsample = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')
        self.conv = ResidualBlock(filters)

    def call(self, inputs, skip_connection=None):
        x = self.upsample(inputs)
        if skip_connection is not None:
            x = tf.concat([x, skip_connection], axis=-1)
        x = self.conv(x)
        return x

class ResNetUNet(Model):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet, self).__init__()
        self.resnet50 = ResNet50(include_top=False, weights=None, input_shape=(256, 256, n_channels))
        
        # Extract intermediate layers
        self.encoder_layers = [
            self.resnet50.get_layer("conv1_relu").output,   # 128x128x64
            self.resnet50.get_layer("conv2_block3_out").output,  # 64x64x256
            self.resnet50.get_layer("conv3_block4_out").output,  # 32x32x512
            self.resnet50.get_layer("conv4_block6_out").output,  # 16x16x1024
            self.resnet50.get_layer("conv5_block3_out").output  # 8x8x2048
        ]
        
        # Create encoder model from ResNet50 backbone
        self.encoder = Model(inputs=self.resnet50.input, outputs=self.encoder_layers)
        
        # Upsample blocks with adjusted filter sizes
        self.up1 = UpSampleBlock(512)
        self.up2 = UpSampleBlock(256)
        self.up3 = UpSampleBlock(128)
        self.up4 = UpSampleBlock(64)
        
        self.outc = layers.Conv2D(n_classes, kernel_size=1)

    def call(self, x):
        # Get encoder outputs
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Decoder with skip connections
        x = self.up1.call(x5, x4)  # 16x16x1024
        x = self.up2.call(x, x3)   # 32x32x512
        x = self.up3.call(x, x2)   # 64x64x256
        x = self.up4.call(x, x1)   # 128x128x128
        
        # Output
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 256x256x64
        output = self.outc(x)
        return output

#------------------------------------------------ UNET ----------------------------------------------------
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




#------------------------------------------------ GAN ----------------------------------------------------
# Discriminator Model
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=[256, 256, 3]),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, x):
        return self.model(x)

# GAN Model
class GAN(Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def train_step(self, blurred_images, sharp_images):
        batch_size = tf.shape(blurred_images)[0]

        # Generate deblurred images from the blurred images
        generated_images = self.generator(blurred_images)

        # Combine generated images with real sharp images
        combined_images = tf.concat([generated_images, sharp_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)

        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(blurred_images))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}
