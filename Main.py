import numpy as np
import matplotlib.pyplot as plt
import MyUtils
import UNet


path_images = "D:\\Work\\Sarbazi\\data\\dataset"
path_masks = "D:\\Work\\Sarbazi\\data\\dataset\\manual segmentation"
batch_size = 4
image_shape = (640, 320)


id_all_data = MyUtils.get_id_all_data(path_masks)
id_training_data, id_testing_data = MyUtils.split_train_test_id(id_all_data,testing_share=0.2)
training_data = MyUtils.DataGenerator(path_images, path_masks, id_training_data, batch_size, image_shape)
testing_data = MyUtils.DataGenerator(path_images, path_masks, id_testing_data, batch_size, image_shape, shuffle=False)

# im, mask = testing_data.__getitem__(0)
# print(im.dtype)
# print(mask.dtype)

# training_data.on_epoch_end()
# for i in range(len(training_data)):
#     print(i)
#     _, _ = training_data.__getitem__(i)
#
# for i in range(len(testing_data)):
#     print(i)
#     _, _ = testing_data.__getitem__(i)

UNetModel = UNet.create_UNet((None,None,1), (None,None,1))
# UNetModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
UNetModel.compile(optimizer="adam", loss=MyUtils.dice_coef_loss, metrics=[MyUtils.dice_coef])

# UNetModel.load_weights('UNetModel_weights.h5')
UNetModel.fit(training_data, validation_data=testing_data, epochs=100, workers=10)
UNetModel.save_weights('UNetModel_weights.h5')

