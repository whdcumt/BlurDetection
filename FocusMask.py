#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# built-in modules
import logging
# Standard modules
import cv2
import numpy
import skimage
import skimage.measure
import skimage.segmentation
# Custom modules
import main
import scripts

logger = logging.getLogger('main')


def get_masks(img, n_seg=250):
    logger.debug('SLIC segmentation initialised')
    segments = skimage.segmentation.slic(img, n_segments=n_seg, compactness=10, sigma=1)
    logger.debug('SLIC segmentation complete')
    logger.debug('contour extraction...')
    masks = [[numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8), None]]
    for region in skimage.measure.regionprops(segments):
        masks.append([masks[0][0].copy(), region.bbox])
        x_min, y_min, x_max, y_max = region.bbox
        masks[-1][0][x_min:x_max, y_min:y_max] = skimage.img_as_ubyte(region.convex_image)
    logger.debug('contours extracted')
    return masks[1:]


def blur_mask_old(img):
    assert isinstance(img, numpy.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    blur_mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)
    for mask, loc in get_masks(img):
        logger.debug('Checking Mask: {0}'.format(numpy.unique(mask)))
        logger.debug('SuperPixel Mask Percentage: {0}%'.format(int((100.0/255.0)*(numpy.sum(mask)/mask.size))))
        img_fft, val, blurry = main.blur_detector(img[loc[0]:loc[2], loc[1]:loc[3]])
        logger.debug('Blurry: {0}'.format(blurry))
        if blurry:
            blur_mask = cv2.add(blur_mask, mask)
    result = numpy.sum(blur_mask)/(255.0*blur_mask.size)
    logger.info('{0}% of input image is blurry'.format(int(100*result)))
    return blur_mask, result


def morphology(msk):
    assert isinstance(msk, numpy.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.erode(msk, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    msk[msk < 128] = 0
    msk[msk > 127] = 255
    return msk


def remove_border(msk, width=50):
    assert isinstance(msk, numpy.ndarray), 'msk must be a numpy array'
    assert msk.ndim == 2, 'msk must be a greyscale image'
    dh, dw = map(lambda i: i//width, msk.shape)
    h, w = msk.shape
    msk[:dh, :] = 255
    msk[h-dh:, :] = 255
    msk[:, :dw] = 255
    msk[:, w-dw:] = 255
    return msk


def blur_mask(img):
    assert isinstance(img, numpy.ndarray), 'img_col must be a numpy array'
    assert img.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img.ndim)
    msk, val, blurry = main.blur_detector(img)
    logger.debug('inverting img_fft')
    msk = cv2.convertScaleAbs(255-(255*msk/numpy.max(msk)))
    msk[msk < 50] = 0
    msk[msk > 127] = 255
    logger.debug('removing border')
    msk = remove_border(msk)
    logger.debug('applying erosion and dilation operators')
    msk = morphology(msk)
    logger.debug('evaluation complete')
    result = numpy.sum(msk)/(255.0*msk.size)
    logger.info('{0}% of input image is blurry'.format(int(100*result)))
    return msk, result, blurry


if __name__ == '__main__':
    img_path = raw_input("Please Enter Image Path: ")
    img = cv2.imread(img_path)
    msk, val = blur_mask(img)
    scripts.display('img', img)
    scripts.display('msk', msk)
    cv2.waitKey(0)
