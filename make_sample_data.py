#!/usr/bin/env python3
"""
make_sample_data.py
--------------------
Generates a tiny synthetic image dataset (colored shapes) so you can
test-run the pipeline immediately without needing a real dataset yet.

Usage:
    python make_sample_data.py
    python run_pipeline.py --data sample_data/shapes --out results.json
"""

import os
import random

from PIL import Image, ImageDraw

OUT_DIR = "sample_data/shapes"
CLASSES = ["circle", "square", "triangle"]
IMAGES_PER_CLASS = 40
IMG_SIZE = 96


def draw_circle(draw, color):
    draw.ellipse([15, 15, 80, 80], fill=color)


def draw_square(draw, color):
    draw.rectangle([15, 15, 80, 80], fill=color)


def draw_triangle(draw, color):
    draw.polygon([(48, 10), (10, 85), (86, 85)], fill=color)


DRAW_FNS = {"circle": draw_circle, "square": draw_square, "triangle": draw_triangle}


def random_color():
    return tuple(random.randint(50, 255) for _ in range(3))


def main():
    random.seed(0)
    for cls in CLASSES:
        cls_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            bg = tuple(random.randint(0, 40) for _ in range(3))
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
            draw = ImageDraw.Draw(img)
            DRAW_FNS[cls](draw, random_color())
            # small random noise/jitter so the task isn't trivially easy
            img = img.rotate(random.uniform(-15, 15), fillcolor=bg)
            img.save(os.path.join(cls_dir, f"{cls}_{i:03d}.png"))

    total = len(CLASSES) * IMAGES_PER_CLASS
    print(f"Generated {total} sample images across {len(CLASSES)} classes in: {OUT_DIR}")


if __name__ == "__main__":
    main()
