import os
from flask import Flask 

app = Flask(__name__)

color = os.environ.get('BG_COLOR')
