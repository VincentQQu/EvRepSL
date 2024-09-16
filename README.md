# EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision

This repository contains the official implementation of the paper **"EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision"**. EvRepSL introduces a novel self-supervised approach for generating event-stream representations, which significantly improves the quality of event-based vision tasks.

## Overview

EvRepSL leverages a two-stage framework for self-supervised learning on event streams. The representation generator **RepGen** learns high-quality representations without requiring labeled data, making it versatile for downstream tasks such as classification and object detection in event-based vision. This repository includes the implementation of the core event representation methods **EvRep** and **EvRepSL**, along with the trained model weights for **RepGen**.

## Repository Structure

- **event_representations.py**: Contains the implementation of the proposed event representation methods, **EvRep** and **EvRepSL**.
- **models.py**: Defines the architecture for **RepGen**, the representation generator trained using self-supervised learning.
- **RepGen.pth**: Pretrained weights for **RepGen** that can be directly used in downstream tasks.
  
## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch numpy
