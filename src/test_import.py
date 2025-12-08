import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.softmax import SoftmaxRegression
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
