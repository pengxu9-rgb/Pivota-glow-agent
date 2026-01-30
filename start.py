from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    sys.path.insert(0, "/app")
    port = int(os.environ.get("PORT") or "8080")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

