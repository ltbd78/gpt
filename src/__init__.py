import os
import sys

a = os.path.realpath(__file__)
b = os.path.dirname(a)
sys.path.append(b) # fixes https://stackoverflow.com/q/77663055/6535624