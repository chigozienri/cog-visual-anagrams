import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, -90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise):
        return TF.rotate(noise, 90, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * -90

        frame = Image.new("RGB", (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(
            theta,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )

        return frame


class Rotate90CCWView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, 90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise):
        return TF.rotate(noise, -90, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 90

        frame = Image.new("RGB", (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(
            theta,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )

        return frame


class Rotate180View(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, 180, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise):
        return TF.rotate(noise, -180, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 180

        frame = Image.new("RGB", (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(
            theta,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255),
        )

        return frame
