from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class BBox:
    '''
    A class for storing bounding box. Each box is represented by two points (x1, y1) and (x2, y2).
    The coordinates are always absolute (ints representing pixels). Width and Height are calculated from the coordinates
    and can be accessed as attributes.
    '''
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float
    confidence: float | None = None

    def __post_init__(self):
        assert self.x1 <= self.x2
        assert self.y1 <= self.y2
        self.x1, self.y1, self.x2, self.y2 = int(self.x1), int(self.y1), int(self.x2), int(self.y2)
        self.w   = self.x2 - self.x1
        self.h   = self.y2 - self.y1
        self.c_x = int(self.x1 + self.w // 2)
        self.c_y = int(self.y1 + self.h // 2)

    @classmethod
    def from_xywh(cls, x, y, w, h) -> BBox:
        return cls(x, y, x + w, y + h)

    @property
    def area(self)->float:
        return self.w * self.h

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y1:self.y2, self.x1:self.x2]


    def __and__(self, other) -> BBox:
        ''' Returns the intersection of two bounding boxes. Can be used like this: box1 & box2
         Confidence of the new box is 0 and must be set manually.'''
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x1 >= x2 or y1 >= y2:
            return BBox(0, 0, 0, 0)

        return BBox(x1, y1, x2, y2)

    def __or__(self, other) -> BBox:
        ''' Returns the union of two bounding boxes. Can be used like this: box1 | box2
         Confidence of the new box is 0 and must be set manually.'''
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BBox(x1, y1, x2, y2)

    def __contains__(self, other) -> bool:
        ''' Returns True if the other box is fully contained in this box
        Can be used like this: if box1 in box2:
        '''
        return self.x1 <= other.x1 and self.y1 <= other.y1 and self.x2 >= other.x2 and self.y2 >= other.y2

    def squarify(self, H, W):
        '''
        Returns a square bbox with max(w,h) size. If the new square bbox is out of image boundaries,
        the CENTER IS SHIFTED to be inside the image.
        '''
        if self.h == self.w:
            return self

        assert (max(self.w, self.h)<= max(H,W)), f'Image size {H}x{W} is smaller than bbox size {self.w}x{self.h}'
        size = min(max(self.w, self.h), min(H,W))
        size = size - size % 2  # Make sure the size is even
        c_x = min(self.c_x + size // 2, W) - size // 2
        c_x = max(c_x - size // 2, 0) + size // 2
        c_y = min(self.c_y + size // 2, H) - size // 2
        c_y = max(c_y - size // 2, 0) + size // 2
        square_bbox = BBox.from_xywh(c_x - size // 2, c_y - size // 2, size, size)
        square_bbox.confidence = self.confidence
        return square_bbox

    def relative_to(self, other: BBox) -> BBox:
        '''
        Returns a new bbox with coordinates relative to the other bbox
        '''
        return BBox(self.x1 - other.x1, self.y1 - other.y1, self.x2 - other.x1, self.y2 - other.y1, self.confidence)

    def draw(self, img: np.ndarray, color=(0, 255, 0), thickness=2):
        return cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def scale(self, factor: float) -> BBox:
        scaled_h = self.h * factor
        scaled_w = self.w * factor

        return BBox(self.c_x - scaled_w // 2, self.c_y - scaled_h // 2, self.c_x + scaled_w // 2, self.c_y + scaled_h // 2, self.confidence)

    def translate(self, dx: int, dy: int) -> BBox:
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy, self.confidence)
    def __repr__(self):
        return f'BBox({self.x1}, {self.y1}, {self.x2}, {self.y2})'