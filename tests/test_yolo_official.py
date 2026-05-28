import os
import sys
import torch
from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26s-seg-pf.pt")


# Run prediction. No prompts required.
results = model.predict("bus.jpg")

# Show results
results[0].show()
