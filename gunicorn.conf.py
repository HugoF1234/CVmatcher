import os

bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 1
worker_class = "sync"
worker_connections = 1000

timeout = 60
keepalive = 5
max_requests = 500
max_requests_jitter = 100

preload_app = True
worker_tmp_dir = "/dev/shm"

accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

worker_timeout = 60
graceful_timeout = 30
