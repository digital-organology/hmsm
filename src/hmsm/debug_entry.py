# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import sys
import hmsm.cli

# This file only serves as an entrypoint for the python debugger,
# as vscodes launch.json config does not yet support cli entrypoints

def main(argv = sys.argv):
    method = argv[1]
    del argv[1]
    if method == "disc2midi":
        hmsm.cli.disc2midi(argv)
    elif method == "disc2roll":
        hmsm.cli.disc2roll(argv)


if __name__ == "__main__":
    main()