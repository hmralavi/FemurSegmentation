"""
 Created By Hamid Alavi on 7/3/2019
"""
import numpy as np
from tensorflow import keras
import cv2
from skimage.transform import resize as img_resize
from skimage import measure, morphology
import nrrd
from pathlib import Path
from keras import backend as K
from UNet import keras_models
import pickle
import bz2


def get_id_all_data(path_data):
    data = Path(path_data).glob("*.femdata")
    ids = []
    for d in data:
        ids.append(int(d.stem))
    return ids


def split_train_test_id(id_all_data, testing_share):
    _id = id_all_data.copy()
    r = np.random.RandomState(seed=1000)
    r.shuffle(_id)
    id_training_data = np.sort(_id[np.floor(len(id_all_data) * testing_share).astype(np.uint16):])
    id_testing_data = np.sort(_id[:np.floor(len(id_all_data) * testing_share).astype(np.uint16)])
    return id_training_data, id_testing_data


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_dataset, batch_size=32, image_resize_to=(512, 512), crop_enabled=False, naugment=1, shuffle=True, load_all=False):
        self.path_dataset = path_dataset
        self.image_ids = get_id_all_data(path_dataset)
        self.batch_size = batch_size
        self.image_resize_to = image_resize_to
        self.crop_enabled = crop_enabled
        self.naugment = naugment
        self.shuffle = shuffle
        self.load_all = load_all
        self.all_data_loaded = {}
        self.random_generator = np.random.RandomState(seed=1000)
        if load_all:
            self._load_all_data()

    def on_epoch_end(self):
        if self.shuffle:
            self.random_generator.shuffle(self.image_ids)

    def __len__(self):
        return int(np.floor(len(self.image_ids)/self.batch_size))
    
    def _load_all_data(self):
        for _id in self.image_ids:
            self.all_data_loaded[_id] = self._load_data(_id)

    def _load_data(self, image_id):
        idstr = str(image_id).zfill(3)
        
#         image_path = Path(self.path_dataset) / Path(idstr+".jpg")
#         mask_path = Path(self.path_dataset) / Path("manual segmentations") / Path(idstr+".nrrd")
#         image = cv2.imread(image_path.__str__())
#         if image.ndim > 2:
#              image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         assert image.ndim == 2, "image id {} must have 2 dims but it has {} dims.".format(idstr, _image.ndim)
        
#         mask, _ = nrrd.read(mask_path.__str__())
#         assert mask.ndim == 3, "mask id {} must have 3 dims but it has {} dims.".format(idstr, mask.ndim)
#         mask = np.transpose(mask, (1, 0, 2)).squeeze(-1)
        
#         assert mask.shape == image.shape, "image shape {} and mask shape {} are not similar for id {}".format(image.shape, mask.shape, idstr)
        
        data_path = Path(self.path_dataset) / Path(idstr+".femdata")
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except:
            with bz2.BZ2File(data_path, 'rb') as f:
                data = pickle.load(f)
        image = data['img']
        mask = data['mask']
        
        images = []
        masks = []
        for iaugment in range(self.naugment):
            _image = image.copy()
            _mask = mask.copy()
            # cropping
            if self.crop_enabled:
                # _mask_locator = locate_femur(_image)
                if iaugment == 0:
                    top, bot, left, right = get_containing_box_corners(_mask, self.image_resize_to, None)
                else:
                    top, bot, left, right = get_containing_box_corners(_mask, self.image_resize_to, self.random_generator)
                imshape = _image.shape

                if bot>imshape[0]:
                    _image = np.pad(_image, ((0,bot-imshape[0]), (0,0)), constant_values=0)
                    _mask  = np.pad(_mask,  ((0,bot-imshape[0]), (0,0)), constant_values=0)
                if top<0:
                    _image = np.pad(_image, ((-top,0), (0,0)), constant_values=0)
                    _mask  = np.pad(_mask,  ((-top,0), (0,0)), constant_values=0)
                    bot += -top
                    top = 0
                if right>imshape[1]:
                    _image = np.pad(_image, ((0,0), (0,right-imshape[1])), constant_values=0)
                    _mask  = np.pad(_mask,  ((0,0), (0,right-imshape[1])), constant_values=0)
                if left<0:
                    _image = np.pad(_image, ((0,0), (-left,0)), constant_values=0)
                    _mask  = np.pad(_mask,  ((0,0), (-left,0)), constant_values=0)
                    right += -left
                    left = 0
                assert _mask.shape == _image.shape, "image shape {} and mask shape {} are not similar for id {} after padding".format(_image.shape, _mask.shape, idstr)

                _image = _image[top:bot, left:right]
                _mask = _mask[top:bot, left:right]
            else:
                rows, cols = _image.shape
                target_ratio = self.image_resize_to[0]/self.image_resize_to[1]
                size_factor = (rows / cols) / target_ratio
                if size_factor > 1:
                    pad_pixels = int(((rows / target_ratio) - cols) // 2)
                    _image = np.pad(_image, ((0, 0), (pad_pixels, pad_pixels)), constant_values=0)
                    _mask = np.pad(_mask, ((0, 0), (pad_pixels, pad_pixels)), constant_values=0)
                elif size_factor < 1:
                    pad_pixels = int(((cols * target_ratio) - rows) // 2)
                    _image = np.pad(_image, ((pad_pixels, pad_pixels), (0, 0)), constant_values=0)
                    _mask = np.pad(_mask, ((pad_pixels, pad_pixels), (0, 0)), constant_values=0)

            # resizing
            _image = img_resize(_image, self.image_resize_to)
            _mask = img_resize(_mask, self.image_resize_to)

            # modifying data type      
            _image = normalize_image_array(_image, np.float32)
            _mask = normalize_image_array(_mask, np.float32)                        

            # expand dims
            _image = np.expand_dims(_image, axis=-1)              
            _mask = np.expand_dims(_mask, axis=-1)  
            
            images.append(_image.copy())
            masks.append(_mask.copy())
        
        return images, masks

    def _get_image_ids(self, batch_index):
        if batch_index == self.__len__() - 1:
            _image_ids = self.image_ids[batch_index * self.batch_size:]
        else:
            _image_ids = self.image_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return _image_ids

    def __getitem__(self, batch_index):
        _image_ids = self._get_image_ids(batch_index)
        images = []
        masks = []
        for _id in _image_ids:
            if self.load_all:
                _image, _mask = self.all_data_loaded[_id]
            else:
                _image, _mask = self._load_data(_id)
            for im in _image:
                images.append(im)
            for ma in _mask:
                masks.append(ma)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks


def dice_coef(y_true, y_pred, smooth=1):
    """
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    """
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def iou_coef(y_true, y_pred, smooth=1):
    """
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    Intersection-Over-Union (IoU, Jaccard Index)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def binary_cross_entropy_and_dice_coef_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + keras.losses.BinaryCrossentropy()(y_true, y_pred)


def get_keras_custom_objects():
    myobjects = {"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef, "binary_cross_entropy_and_dice_coef_loss": binary_cross_entropy_and_dice_coef_loss}
    return myobjects


def normalize_image_array(img, output_dtype):
    im = img.copy()
    im = im.astype(np.float32)
    maxval = np.max(im)
    if maxval>0:
        im = im/np.max(im)
    if output_dtype == np.uint8:
        im = (im * (2**8-1)).astype(np.uint8)
    elif output_dtype == np.uint16:
        im = (im * (2**16-1)).astype(np.uint16)
    elif output_dtype == np.bool_:
        im = (im>0).astype(np.bool_)
    return im


def get_containing_box_corners(mask, target_shape, rndgen=None):
    row, col = np.where(mask > 0)
    top = np.min(row)
    bot = np.max(row)
    left = np.min(col)
    right = np.max(col)

    target_ratio = target_shape[0]/target_shape[1]
    size_factor = ((bot-top) / (right-left)) / target_ratio
    if size_factor > 1:
        pad_pixels = ((bot-top) / target_ratio - (right-left)) // 2
        right += pad_pixels
        left -= pad_pixels
    elif size_factor < 1:
        pad_pixels = ((right-left) * target_ratio - (bot-top)) // 2
        bot += pad_pixels
        top -= pad_pixels
    if rndgen:
        scale = rndgen.randn()/4
        width_pad = int((right-left)*abs(scale))*2
        height_pad = int((bot-top)*abs(scale))*2
        right += width_pad//2
        left -= width_pad//2
        top -= height_pad//2
        bot += height_pad//2
        xtranslate = rndgen.randint(-width_pad//2, width_pad//2+1)
        ytranslate = rndgen.randint(-height_pad//2, height_pad//2+1)
        top += ytranslate
        bot += ytranslate
        left += xtranslate
        right += xtranslate
    return int(top), int(bot), int(left), int(right)


def keep_biggest_cluster_only(mask):
    # assuming mask is a binary image
    # label and calculate parameters for every cluster in mask
    for i in range(mask.shape[0]):
        labelled = measure.label(mask[i,...])
        rp = measure.regionprops(labelled)

        # get size of largest cluster
        size = max([j.area for j in rp])

        # remove everything smaller than largest
        mask[i,...] = morphology.remove_small_objects(mask[i,...], min_size=size-1)
    

def locate_femur(im):
    unet = keras_models.load_model("UNetModel_locator.h5", custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})

    rows, cols = im.shape
    pad_rows = 0
    pad_cols = 0
    if rows > cols:
        pad_cols = (rows - cols) // 2
        im = np.pad(im, ((0, 0), (pad_cols, pad_cols)), constant_values=0)
    elif rows < cols:
        pad_rows = (cols - rows) // 2
        im = np.pad(im, ((pad_rows, pad_rows), (0, 0)), constant_values=0)

    rows, cols = im.shape
    im = img_resize(im, unet.input_shape[1:3])
    im = np.expand_dims(im, axis=0)
    im = np.expand_dims(im, axis=-1)
    im = im.astype(np.float32)
    im = im / np.max(im)
    assert im.shape[1:] == unet.input_shape[1:], "image shape is not similar as unet locator input"
    predict_segment = unet.predict(im, verbose=0)
    predict_segment = (predict_segment > 0.5).astype(np.bool_)[0, :, :, 0]
    predict_segment = img_resize(predict_segment, (rows, cols))
    predict_segment = predict_segment.astype("uint8")
    predict_segment = predict_segment[pad_rows:rows-pad_rows, pad_cols:cols-pad_cols]
    return predict_segment