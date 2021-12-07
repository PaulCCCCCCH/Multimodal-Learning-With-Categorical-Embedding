import torch
import numpy as np
# Semantic Segmentation
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from PIL import Image
# OCR
import pytesseract
import re
from english_words import english_words_set


class BaseCategorizer:
    pass


class Categorizer(BaseCategorizer):

    def __init__(self, pixel_group=100, min_color=5, max_group=12, cate_feat_dim=152, select=None):
        # cate_feat_dim: Dimension of output category feature vector. 150 for semantic segmentation + 2 extra

        assert pixel_group > 0

        self.pixel_group = pixel_group
        self.min_color = min_color
        self.max_group = max_group
        self.cate_feat_dim = cate_feat_dim
        self.select = select

        # Network Builders
        self.net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='W:/models/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        self.net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='W:/models/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(
            self.net_encoder, self.net_decoder, crit)
        self.segmentation_module.eval()
        self.segmentation_module.cuda()

    def categorize(self, data):
        """Returns a category vector of the given data

        Args:
            data ([dict]): A dictionary obtained from dataloader. Batch size should be 1

        Returns:
            a numpy array of shape [DIM]

        """
        result = np.zeros(self.cate_feat_dim)
        result[0:1] = self._get_feat_is_raw(data)
        result[1:2] = self._get_feat_has_text(data)
        result[2:] = self._get_feat_semantic_seg(data)

        return result

    def _get_feat_semantic_seg(self, data):
        img_data = data['image'][0][None].cuda()    # img_data (1, 3, 224, 224)
        output_size = img_data.shape[2:]            # output size (224, 224)
        seg_out = self.segmentation_module(
            {'img_data': img_data}, segSize=output_size)[0]  # seg_out (150, 224, 224)

        pred = torch.sum(seg_out, dim=(1, 2)).cpu().detach().numpy()
        # pred: (150) vector, not normalized
        pred = pred[self.select]
        return pred

    def _get_feat_is_raw(self, data):
        # data: a torch tensor of shape (3, 224, 224)
        image_arr = data['image'][0]
        height = image_arr.shape[1]
        width = image_arr.shape[2]
        group_count = 0  # How many groups of adjacent pixels have the exact same color?
        total_count = 0

        # Find groups in lines
        for row in range(height):
            for col in range(0, width - self.pixel_group + 1, self.pixel_group):
                group = set()
                different = False
                for i in range(col, col + self.pixel_group):
                    pix = tuple(image_arr[:, row, i].cpu().numpy())
                    group.add(pix)
                    if len(group) > self.min_color:
                        different = True
                        break
                if not different:
                    group_count += 1
                total_count += 1

        for col in range(width):
            for row in range(0, height - self.pixel_group + 1, self.pixel_group):
                group = set()
                different = False
                for i in range(row, row + self.pixel_group):
                    pix = tuple(image_arr[:, i, col].cpu().numpy())
                    group.add(pix)
                    if len(group) > self.min_color:
                        different = True
                        break
                if not different:
                    group_count += 1
                total_count += 1

        # TODO: Try more advanced criteria
        if group_count > self.max_group:
            # Too many groups have exactly the same color. Must be processed image
            return np.array([0])
        return np.array([1])

    def _get_feat_has_text(self, data):
        image_path = data['path_image'][0]
        out = pytesseract.image_to_string(image_path)
        out = re.sub("[^A-z0-9,. ]", "", out)
        tokens = re.sub("[^A-z ]", "", out).split(' ')
        if re.search('[A-z]+', out):
            valid = sum(
                [1 if token in english_words_set else 0 for token in tokens])
            total = len(tokens)

            # TODO: Try more advanced criteria
            if valid > 10 or valid > 0.5 * total:
                return np.array([1])
        return np.array([0])
        #     return np.array([valid])
        # return np.array([0])
