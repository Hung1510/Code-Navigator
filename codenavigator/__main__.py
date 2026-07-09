"""Enable `python -m codenavigator ...` (used by the desktop backend)."""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
