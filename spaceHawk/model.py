import tensorflow as tf
from tensorflow.keras import layers

class DoubleConv(tf.keras.Model):
    def __init__(self, out_channels):
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
    def __init__(self, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2)),
            DoubleConv(out_channels)
        ])

    def call(self, x):
        return self.maxpool_conv(x)

class FPN(tf.keras.Model):
    def __init__(self, n_channels, n_classes):
        super(FPN, self).__init__()
        self.inc = DoubleConv(64)
        self.down1 = Down(128)
        self.down2 = Down(256)
        self.down3 = Down(512)
        self.down4 = Down(512)

        # Lateral layers
        self.latlayer1 = layers.Conv2D(256, kernel_size=1, padding='same')
        self.latlayer2 = layers.Conv2D(256, kernel_size=1, padding='same')
        self.latlayer3 = layers.Conv2D(256, kernel_size=1, padding='same')
        self.latlayer4 = layers.Conv2D(256, kernel_size=1, padding='same')

        # Smooth layers
        self.smooth1 = layers.Conv2D(256, kernel_size=3, padding='same')
        self.smooth2 = layers.Conv2D(256, kernel_size=3, padding='same')
        self.smooth3 = layers.Conv2D(256, kernel_size=3, padding='same')

        self.outc = layers.Conv2D(n_classes, kernel_size=1)

    def call(self, x):
        # Bottom-up pathway
        c1 = self.inc(x)  # 256x256x64
        c2 = self.down1(c1)  # 128x128x128
        c3 = self.down2(c2)  # 64x64x256
        c4 = self.down3(c3)  # 32x32x512
        c5 = self.down4(c4)  # 16x16x512

        # Top-down pathway and lateral connections
        p5 = self.latlayer1(c5)  # 16x16x256
        p4 = self.latlayer2(c4) + tf.image.resize(p5, size=(tf.shape(c4)[1], tf.shape(c4)[2]))  # 32x32x256
        p4 = self.smooth1(p4)
        p3 = self.latlayer3(c3) + tf.image.resize(p4, size=(tf.shape(c3)[1], tf.shape(c3)[2]))  # 64x64x256
        p3 = self.smooth2(p3)
        p2 = self.latlayer4(c2) + tf.image.resize(p3, size=(tf.shape(c2)[1], tf.shape(c2)[2]))  # 128x128x256
        p2 = self.smooth3(p2)

        p1 = tf.image.resize(p2, size=(tf.shape(c1)[1], tf.shape(c1)[2]))  # 256x256x256

        results = self.outc(p1)  # 256x256x<n_classes>
        return results