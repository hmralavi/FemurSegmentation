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
    def __init__(self, path_images, path_masks, ids, batch_size=32, image_resize_to=(480, 640), shuffle=True):
        self.path_images = path_images
        self.path_masks = path_masks
        self.image_ids = ids.copy()
        self.batch_size = batch_size
        self.image_resize_to = image_resize_to
        self.shuffle = shuffle
        self.random_generator = np.random.RandomState(seed=1000)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.random_generator.shuffle(self.image_ids)

    def __len__(self):
        return int(np.floor(len(self.image_ids)/self.batch_size))

    def _load_data(self, image_id):
        idstr = str(image_id).zfill(3)
        image_path = Path(self.path_images) / Path(idstr+".tif")
        mask_path = Path(self.path_masks) / Path(idstr+".nrrd")
                
        # _image = np.array(PILImage.open(image_path.__str__()))
        _image = cv2.imread(image_path.__str__())
        if _image.ndim > 2:
            _image = _image[:, :, 0]
        # _image = img_io.imread(image_path)
        assert _image.ndim == 2, "image id {} must have 2 dims but it has {} dims.".format(idstr, _image.ndim)
        
        _mask, _ = nrrd.read(mask_path.__str__())
        assert _mask.ndim == 3, "mask id {} must have 3 dims but it has {} dims.".format(idstr, _mask.ndim)
        _mask = np.transpose(_mask, (1, 0, 2)).squeeze(-1)
        
        assert _mask.shape == _image.shape, "image shape {} and mask shape {} are not similar for id {}".format(_image.shape, _mask.shape, idstr)
        
        # cropping
        image_shape = _image.shape
        cropwidth = np.ceil(image_shape[1] * 0.4)
        cropheight = np.ceil(self.image_resize_to[0] * cropwidth / self.image_resize_to[1])
        cropwidth = int(cropwidth)
        cropheight = int(cropheight)
        
        padheight = cropheight - image_shape[0]
        if padheight>0:
            _image = np.pad(_image, ((padheight//2+1,), (0,)), constant_values=0)
            _mask = np.pad(_mask, ((padheight//2+1,), (0,)), constant_values=0)
            assert _mask.shape == _image.shape, "image shape {} and mask shape {} are not similar for id {} after padding".format(_image.shape, _mask.shape, idstr)
            
        image_shape = _image.shape
        
        assert image_shape[0]-cropheight>=0, "heigth cropping failed for image id {}".format(idstr)
        assert image_shape[1]-cropwidth>=0, "width cropping failed for image id {}".format(idstr)
                      
        _image = _image[image_shape[0]-cropheight:, image_shape[1]-cropwidth:]
        _mask = _mask[image_shape[0]-cropheight:, image_shape[1]-cropwidth:]            
        
        # resizing        
        _image = img_resize(_image, self.image_resize_to)
        _mask = img_resize(_mask, self.image_resize_to)
        
        # modifying data type      
        _image = normalize_image_array(_image, np.float32)
        _mask = normalize_image_array(_mask, np.bool_)                        
        
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
            _image, _mask = self._load_data(_id)
            images.append(_image)
            masks.append(_mask)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks


def dice_coef_(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_pred = K.greater_equal(y_pred,0.5)
    y_pred = K.cast(y_pred, dtype=K.floatx())
    y_true = K.cast(y_true, dtype=K.floatx())
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef(y_true, y_pred, smooth=100):
    y_pred = K.greater_equal(y_pred, 0.5)
    y_pred = K.cast(y_pred, dtype=K.floatx())
    y_true = K.cast(y_true, dtype=K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
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