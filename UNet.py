"""
 Created By Hamid Alavi on 7/3/2019
"""
from tensorflow import keras
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, UpSampling2D
import keras.models as keras_models



def create_UNet_locator(input_shape, output_shape):
    filters = [4, 8, 16, 32, 64, 128]
    input_layer = Input(input_shape)

    c1, p1 = down_block(input_layer, filters[0])
    c2, p2 = down_block(p1, filters[1])
    c3, p3 = down_block(p2, filters[2])
    c4, p4 = down_block(p3, filters[3])
    c5, p5 = down_block(p4, filters[4])
    
    bn = bottleneck_block(p5, filters[5])

    u1, _ = up_block(bn, c5, filters[4])  
    u2, _ = up_block(u1, c4, filters[3])
    u3, _ = up_block(u2, c3, filters[2])
    u4, _ = up_block(u3, c2, filters[1])
    _, u5 = up_block(u4, c1, filters[0])

    output_layer = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation='sigmoid')(u5)
    return keras_models.Model(input_layer, output_layer)


def create_UNet_accurate(input_shape, output_shape):
    filters = [4, 8, 16, 32, 64, 128]
    input_layer = Input(input_shape)

    c1, p1 = down_block(input_layer, filters[0])
    c2, p2 = down_block(p1, filters[1])
    c3, p3 = down_block(p2, filters[2])
    c4, p4 = down_block(p3, filters[3])
    c5, p5 = down_block(p4, filters[4])
    
    bn = bottleneck_block(p5, filters[5])

    u1, _ = up_block(bn, c5, filters[4])  
    u2, _ = up_block(u1, c4, filters[3])
    u3, _ = up_block(u2, c3, filters[2])
    u4, _ = up_block(u3, c2, filters[1])
    _, u5 = up_block(u4, c1, filters[0])

    output_layer = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation='sigmoid')(u5)
    return keras_models.Model(input_layer, output_layer)


def down_block(x, filters):
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c)
    p = MaxPooling2D((2, 2), (2, 2))(c)
    return c, p


def bottleneck_block(x, filters):
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c)
    us = UpSampling2D((2, 2))(c)
    return us


def up_block(x, skip_layer, filters):
    concat = Concatenate()([x, skip_layer])
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    c = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(c)
    us = UpSampling2D((2, 2))(c)
    return us, c

if __name__ == '__main__':
    UNetModel = create_UNet((480,840, 1), (480,840, 4))
    UNetModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    UNetModel.summary()