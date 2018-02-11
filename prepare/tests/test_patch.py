import unittest
import numpy as np
from ..patch import pad_for_patching
from ..patch import patches_generator


class TestPadForPatching(unittest.TestCase):
    def test_pad_for_patching_1280_720(self):
        image = np.zeros((1280, 720, 3))
        patch_size = (256, 256, 3)
        expected_shape = (1280, 768, 3)
        actual_shape = pad_for_patching(image, patch_size, 0).shape
        self.assertEqual(expected_shape, actual_shape)

    def test_pad_for_patching_640_360(self):
        image = np.zeros((640, 360, 3))
        patch_size = (256, 256, 3)
        expected_shape = (768, 512, 3)
        actual_shape = pad_for_patching(image, patch_size, 0).shape
        self.assertEqual(expected_shape, actual_shape)

    def test_pad_for_patching_500_500(self):
        image = np.zeros((500, 500, 3))
        patch_size = (256, 256, 3)
        expected_shape = (512, 512, 3)
        actual_shape = pad_for_patching(image, patch_size, 0).shape
        self.assertEqual(expected_shape, actual_shape)


class TestPatchesGEnerator(unittest.TestCase):
    def test_patches_1280_768_overlap_0(self):
        image = np.zeros((1280, 768, 3))
        patch_size = (256, 256, 3)
        overlap = 0

        actual_patches = list(patches_generator(image, patch_size, overlap, 0))
        self.assertEqual(15, len(actual_patches))
        patch_shapes = set()
        for patch in actual_patches:
            patch_shapes.add(patch.shape)
        self.assertEqual(1, len(patch_shapes), msg='Patch shapes are not equal')

    def test_patches_1280_768_overlap_05(self):
        image = np.zeros((1280, 768, 3))
        patch_size = (256, 256, 3)
        overlap = 0.5

        actual_patches = list(patches_generator(image, patch_size, overlap, 0))
        self.assertEqual(45, len(actual_patches))
        patch_shapes = set()
        for patch in actual_patches:
            patch_shapes.add(patch.shape)
        self.assertEqual(1, len(patch_shapes), msg='Patch shapes are not equal')
