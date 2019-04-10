
from mlearn import mLearn
import sys

if __name__ == "__main__":
    nflod = int(sys.argv[1])

    mode = mLearn()
    mode.execute(nflod)