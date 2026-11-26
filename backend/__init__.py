"""
GROUND LENS Backend Package Initialization
"""

import warnings

# Cleanly suppress Python 3.9 EOL / LibreSSL notifications from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
