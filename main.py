
from mlearn import mLearn
import sys

from lgbbayes import LGBBayes
if __name__ == "__main__":
    nflod = int(sys.argv[1])

    # mode = mLearn()
    # mode.execute(nflod)

    bayesMode = LGBBayes()
    bayesMode.execute(nflod)