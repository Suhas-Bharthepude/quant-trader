# conftest.py  (repository root)
#
# pytest automatically loads every conftest.py it finds, starting from the
# directory where it is invoked and walking up to the rootdir.  Placing this
# file at the repo root means it runs for every test in the project.

import sys
from pathlib import Path

# Path(__file__) is the absolute path to this conftest.py file.
# .parent gives the directory that contains it — the repository root.
# str() converts it to a plain string because sys.path holds strings.
project_root = str(Path(__file__).parent)

# sys.path is the list of directories Python searches when resolving imports.
# insert(0, ...) prepends the repo root so it is checked before site-packages,
# which ensures `from src.brokers...` resolves to our local src/ tree rather
# than any installed package with the same name.
sys.path.insert(0, project_root)
