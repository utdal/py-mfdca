try:
    import sys
    if sys.platform != "win32":
        from . import hmm_tools
except ImportError:
    hmm_tools = None