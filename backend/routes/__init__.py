"""
Routes package initialization for AI Insight Hub backend.
"""
from flask import Blueprint

# Main API Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

from . import health
from . import r_engine
from . import datasets
from . import regression

