import tensorflow as tf


class ConvBlock:
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        K=1,
        backbone="residual",
    ):
        super(ConvBlock, self).__init__()

        self.f = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=kernel_size,
            strides=stride,
            padding="SAME",
            use_bias=bias,
            activation=None,
        )
        self.g = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            activation=None,
        )
        self.K = K
        self.backbone = backbone

        self.bn_out = tf.keras.layers.BatchNormalization()
        self.bn_f1 = tf.keras.layers.BatchNormalization()
        self.c_out = c_out

    def __call__(self, x):
        f = self.f(tf.keras.layers.Activation("relu")(self.bn_f1(x)))
        h = f

        if self.backbone == "cnn":
            bn_g = tf.keras.layers.BatchNormalization()
            h = self.g(tf.keras.layers.Activation("relu")(bn_g(h)))
            # # the repo missed these layers!
            # h = self.bn_out(h)
            # h = tf.keras.layers.Activation("relu")(h)
        elif self.backbone == "residual":
            for k in range(self.K):
                bn_g = tf.keras.layers.BatchNormalization()
                g = tf.keras.layers.Conv2D(
                    self.c_out, kernel_size=3, strides=1, padding="SAME", use_bias=False
                )
                h = h + g(tf.keras.layers.Activation("relu")(bn_g(h)))

            h = self.bn_out(h)
            h = tf.keras.layers.Activation("relu")(h)
        else:
            h0 = h  # f #h
            # f = h

            bn_g = tf.keras.layers.BatchNormalization()
            g = self.g(tf.keras.layers.Activation("relu")(bn_g(h)))
            g1 = g
            # g  = tf.keras.layers.Activation('relu')( self.bng(self.convg(h)) )
            # g1 = tf.keras.layers.Activation('relu')( self.bng1(self.convg1(h)) )

            dt = 0.2
            dx = 1.0
            dy = 1.0

            Dx = 1.0
            Dy = 1.0
            # Dx  = tf.keras.layers.Activation('relu')( self.bnDx(self.convDx(h)) )
            # Dy  = tf.keras.layers.Activation('relu')( self.bnDy(self.convDy(h)) )

            ux = (1.0 / (2 * dx)) * (tf.roll(g, 1, axis=1) - tf.roll(g, -1, axis=1))
            vy = (1.0 / (2 * dy)) * (tf.roll(g1, 1, axis=2) - tf.roll(g1, -1, axis=2))

            Ax = g * (dt / dx)
            Ay = g1 * (dt / dy)
            Bx = Dx * (dt / (dx * dx))
            By = Dy * (dt / (dy * dy))
            E = (ux + vy) * dt

            D = 1.0 / (1 + 2 * Bx + 2 * By)

            for k in range(self.K):
                prev_h = h

                h = D * (
                    (1 - 2 * Bx - 2 * By) * h0
                    - 2 * E * h
                    + (-Ax + 2 * Bx) * tf.roll(h, 1, axis=1)
                    + (Ax + 2 * Bx) * tf.roll(h, -1, axis=1)
                    + (-Ay + 2 * By) * tf.roll(h, 1, axis=2)
                    + (Ay + 2 * By) * tf.roll(h, -1, axis=2)
                    + 2 * dt * f
                )
                h0 = prev_h

            h = self.bn_out(h)
            h = tf.keras.layers.Activation("relu")(h)

        return h


def net(backbone="residual", K=5):

    inp = tf.keras.Input((28, 28, 1))

    conv = ConvBlock(1, 1, 3, K=K, backbone=backbone)
    x = conv(inp)
    f1 = x
    x = tf.keras.layers.Activation("relu")(x)

    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
    x = avg_pool(x)
    f2 = x
    flatten = tf.keras.layers.Flatten()
    x = flatten(x)

    fc = tf.keras.layers.Dense(10)
    out = fc(x)

    return tf.keras.Model(inp, [out, f1, f2])
