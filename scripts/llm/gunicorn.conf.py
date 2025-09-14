# gunicorn.conf.py

# The host and port to bind to
bind = "0.0.0.0:5000"

# Number of worker processes. A good starting point is (2 x $num_cores) + 1
workers = 1

# The type of worker class
worker_class = "sync"

# Timeout for handling a request (in seconds)
# This is VERY important. If your model takes a long time to generate text,
# the default 30s timeout will kill the request. Set it to a higher value.
timeout = 3000  # 50 minutes

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr

# Note : use code 
# # Make sure (venv) is active
# python -m gunicorn --config gunicorn.conf.py llm_flask:app
