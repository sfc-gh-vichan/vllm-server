import logging
import uvicorn

from app import app  # noqa
from args import args

if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=args.port,
        log_level=logging.INFO,
    )
