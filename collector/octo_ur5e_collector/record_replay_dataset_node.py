"""Compatibility entry point for replay capture."""
from .collector_cli import main

if __name__ == "__main__": main(["replay",*__import__("sys").argv[1:]])
