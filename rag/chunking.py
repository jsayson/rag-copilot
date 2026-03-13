import numpy as np
import matplotlib.pyplot as plt
import math
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

class chunker():
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"current text: {self.text}"
    
