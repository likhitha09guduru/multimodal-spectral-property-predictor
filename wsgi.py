"""
Production WSGI entry point.

Run with:
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 wsgi:application

-w 2, not more: each worker loads its own copy of the model into memory, so
worker count should be sized against available RAM, not just CPU cores.
--timeout 120: generous headroom for cold model load on the first request
into a fresh worker.
"""

from app import application

if __name__ == "__main__":
    application.run()
