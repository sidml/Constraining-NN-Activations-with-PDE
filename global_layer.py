import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    Activation,
    Lambda,
)


def create_conv2d(filters, ksize, stride):
    # we use depthwise conv with groups=filters
    return Conv2D(
        filters,
        (ksize, ksize),
        strides=stride,
        use_bias=False,
        padding="same",
        groups=filters,
    )


class GlobalFeatureBlock_Diffusion:
    expansion: int = 1

    def __init__(self, planes, args, block_num):
        super(GlobalFeatureBlock_Diffusion, self).__init__()

        self.block_num = block_num
        K = args.get("K", 10)

        cDx = args.get("cDx", 1.0)
        cDy = args.get("cDy", 1.0)
        dx = args.get("dx", 1)
        dy = args.get("dy", 1)
        dt = args.get("dt", 0.2)

        constant_Dxy = args.get("constant_Dxy", "False")
        non_linear_Dxy = args.get("non_linear_Dxy", "False")
        init_h0_h = args.get("init_h0_h", False)
        no_f = args.get("no_f", False)
        disable_advection = args["disable_advection"]
        disable_diffusion = args.get("disable_diffusion", "False")

        stride = args.get("stride", 1)
        in_chs = args.get("in_chs", planes)
        out_chs = args.get("out_chs", planes)

        print("non_linear_Dxy", non_linear_Dxy)

        self.K = K

        self.bn_out = BatchNormalization(axis=3)

        self.init_h0_h = init_h0_h
        self.dx = dx
        self.dy = dy
        self.cDx = cDx
        self.cDy = cDy
        self.no_f = no_f
        self.dt = dt
        self.constant_Dxy = constant_Dxy
        self.non_linear_Dxy = non_linear_Dxy
        self.disable_advection = disable_advection
        self.disable_diffusion = disable_diffusion

        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.planes = planes

        if disable_advection == "False":
            self.convg = create_conv2d(planes, ksize=3, stride=1)
            self.convg1 = create_conv2d(planes, ksize=3, stride=1)

            self.bng = BatchNormalization(axis=3)
            self.bng1 = BatchNormalization(axis=3)

        if (constant_Dxy == "False") and (non_linear_Dxy == "False") and (disable_diffusion == "False"):
            self.convDx = create_conv2d(planes, ksize=3, stride=1)
            self.convDy = create_conv2d(planes, ksize=3, stride=1)
            self.bnDx = BatchNormalization(axis=3)
            self.bnDy = BatchNormalization(axis=3)

    def __call__(self, s0):
        s0 = Lambda(lambda x: x, name=f"I_block{self.block_num}")(s0)

        f = s0
        # identity
        h = f

        if (self.stride != 1) or (self.in_chs != self.out_chs):
            f = h
        residual = f

        if self.init_h0_h:
            h0 = h
        else:
            h0 = f

        g0 = h

        if self.disable_advection == "False":
            g = Activation("relu", name=f"g_block{self.block_num}")(
                self.bng(self.convg(g0))
            )
            g1 = Activation("relu", name=f"g1_block{self.block_num}")(
                self.bng1(self.convg1(g0))
            )
        else:
            g, g1 = 0, 0

        dt = self.dt
        dx = self.dx
        dy = self.dy
        if self.disable_diffusion=="True":
            Dx = 0.
            Dy = 0.
        elif self.constant_Dxy=="True":
            # Dx = self.cDx
            # Dy = self.cDy
            Dx = Lambda(lambda x: x, name=f"Dx_block{self.block_num}")(self.cDx)
            Dy = Lambda(lambda x: x, name=f"Dy_block{self.block_num}")(self.cDy)
        else:
            Dx = Activation("relu", name=f"Dx_block{self.block_num}")(
                self.bnDx(self.convDx(h))
            )
            Dy = Activation("relu", name=f"Dy_block{self.block_num}")(
                self.bnDy(self.convDy(h))
            )

        if self.disable_advection == "False":
            ux = (1.0 / (2 * dx)) * (tf.roll(g, dx, axis=1) - tf.roll(g, -dx, axis=1))
            ux = Lambda(lambda x: x, name=f"ux_block{self.block_num}")(ux)
            vy = (1.0 / (2 * dy)) * (tf.roll(g1, dy, axis=2) - tf.roll(g1, -dy, axis=2))
            vy = Lambda(lambda x: x, name=f"vy_block{self.block_num}")(vy)
        else:
            ux = 0
            vy = 0

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
            )
            if self.no_f == False:
                h = h + D * 2 * dt * f

            h0 = prev_h

        h = self.bn_out(h)
        h = Activation("relu")(h)
        h = Lambda(lambda x: x, name=f"h_block{self.block_num}")(h)

        return h