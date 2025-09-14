"""
Test the LLM server to check for responses
"""
from flask import Flask, request, jsonify
from llama_cpp import Llama
import argparse
import os

import requests

