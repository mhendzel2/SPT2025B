#!/bin/bash
pip install -r requirements.txt
python -m pytest tests/test_app_logic.py
