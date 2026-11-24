# Resolve the official venv's wheel .pth entries at the unpacked root path.
import os
import site
site.addsitedir(os.environ['PEARL_TEST_SITE'])
