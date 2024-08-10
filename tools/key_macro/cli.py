import argparse
from argparse import RawTextHelpFormatter

from . import core


def main():
    description = f"key_macro custom settings:\n{core.read_custom_template()}"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("key", type=str, help="custom key")
    parser.add_argument("value", type=str, help="custom value")
    args = parser.parse_args()
    core.update_custom_template(args.key, args.value)


if __name__ == "__main__":
    main()
