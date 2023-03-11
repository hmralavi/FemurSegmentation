"""
 Created By Hamid Alavi on 7/3/2019
"""
import numpy as np
from tensorflow import keras
# from PIL import Image as PILImage
import cv2
from skimage.transform import resize as img_resize
import nrrd
from pathlib import Path
from keras import backend as K


def get_id_all_data(path_masks):
    masks = Path(path_masks).glob("*.nrrd")
    ids = []
    for mask in masks:
        ids.append(int(mask.stem))
    return ids


def split_train_test_id(id_all_data, testing_share):
    _id = id_all_data.copy()
    r = np.random.RandomState(seed=1000)
    r.shuffle(_id)
    id_training_data = np.sort(_id[np.floor(len(id_all_data) * testing_share).astype(np.uint16):])
    id_testing_data = np.sort(_id[:np.floor(len(id_all_data) * testing_share).astype(np.uint16)])
    return id_training_data, id_testing_data


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_images, path_masks, ids, batch_size=32, image_resize_to=(512, 512), crop_enabled=False, shuffle=True, load_all=False):
        self.path_images = path_images
        self.path_masks = path_masks
        self.image_ids = ids.copy()
        self.batch_size = batch_size
        self.image_resize_to = image_resize_to
        self.crop_enabled = crop_enabled
        self.shuffle = shuffle
        self.load_all = load_all
        self.all_data_loaded = {}
        if load_all:
            self._load_all_data()
        self.random_generator = np.random.RandomState(seed=1000)
        self.on_epoch_end()

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
        image_path = Path(self.path_images) / Path(idstr+".tif")
        mask_path = Path(self.path_masks) / Path(idstr+".nrrd")
                
        _image = cv2.imread(image_path.__str__())
        if _image.ndim > 2:
            _image = _image[:, :, 0]
        assert _image.ndim == 2, "image id {} must have 2 dims but it has {} dims.".format(idstr, _image.ndim)
        
        _mask, _ = nrrd.read(mask_path.__str__())
        assert _mask.ndim == 3, "mask id {} must have 3 dims but it has {} dims.".format(idstr, _mask.ndim)
        _mask = np.transpose(_mask, (1, 0, 2)).squeeze(-1)
        
        assert _mask.shape == _image.shape, "image shape {} and mask shape {} are not similar for id {}".format(_image.shape, _mask.shape, idstr)
        
        # cropping
        if self.crop_enabled:
            top, bot, left, right = get_containing_box_corners(_mask, self.image_resize_to)
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
            if rows > cols:
                pad_pixels = (rows - cols) // 2
                _image = np.pad(_image, ((0, 0), (pad_pixels, pad_pixels)), constant_values=0)
                _mask = np.pad(_mask, ((0, 0), (pad_pixels, pad_pixels)), constant_values=0)
            elif rows < cols:
                pad_pixels = (cols - rows) // 2
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
        
        return _image, _mask

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
            images.append(_image)
            masks.append(_mask)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def normalize_image_array(img, output_dtype):
    im = img.copy()
    im = im.astype(np.float32)
    im = im/np.max(im)
    if output_dtype == np.uint8:
        im = (im * (2**8-1)).astype(np.uint8)
    elif output_dtype == np.uint16:
        im = (im * (2**16-1)).astype(np.uint16)
    elif output_dtype == np.bool_:
        im = (im>0).astype(np.bool_)
    return im


def get_containing_box_corners(mask, target_shape):
    row, col = np.where(mask > 0)
    top = np.min(row)
    bot = np.max(row)
    left = np.min(col)
    right = np.max(col)
    top -= 100
    left -= 100
    right += 100
    bot = (right-left)*(target_shape[0]/target_shape[1]) + top
    bot = np.ceil(bot)
    return int(top), int(bot), int(left), int(right)
    
