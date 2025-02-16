# jarvis-health

## Overview

Jarvis Health is a platform that helps you manage your health.

## Features


## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
fastapi dev main.py
```

```bash
curl -X POST "http://127.0.0.1:8000/optimize" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How can I improve my endurance for long-range missions?",
       "soldier_data": {
         "steps": 8000,
         "heart_rate": 70,
         "sleep_hours": 7
       }
     }'
```
