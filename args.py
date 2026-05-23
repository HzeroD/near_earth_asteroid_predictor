import argparse, string, random
from pathlib import Path

# parser = argparse.ArgumentParser()

# parser.add_argument("-p", "--place", type=str)
# parser.add_argument("-c", "--city", default="towns",type=str)
# parser.add_argument("-v", "--verb", type=str)
# parser.add_argument("-r", action="store_true", default=False)

# args = parser.parse_args()

# print(f"Of all the {args.place}, out of all of the {args.city} in the world,")
# print(f"she just had to {args.verb} into mine")

# for entry in Path("./artifacts/data").iterdir():

#     print(entry)
#     print(entry.name)

# parser2 = argparse.ArgumentParser()

# parser2.add_argument("--name", action="store")
# parser2.add_argument("--pi", action="store_const", const=3.14)
# parser2.add_argument("--is-valid", action="store_true", default=False)
# parser2.add_argument("--is-invalid", action="store_false")
# parser2.add_argument("--item", action="append")
# parser2.add_argument("--repeated", action="append_const", const=42)
# parser2.add_argument("--add-one", action="count")
# parser2.add_argument("--version", action="version", version="%(prog)s v0.1.0")

# args2 = parser2.parse_args()

# print(args2)

import tempfile

with tempfile.TemporaryDirectory() as tempf:
    print(Path(tempf))
