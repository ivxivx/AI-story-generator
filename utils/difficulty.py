from enum import Enum

class Difficulty(Enum):
    CHILD = "child"
    PRIMARY = "primary"
    UNIVERSITY = "university"
    PROFESSIONAL = "professional"


WORD_COUNTS = {
    Difficulty.CHILD: [20, 60],
    Difficulty.PRIMARY: [60, 100],
    Difficulty.UNIVERSITY: [100, 300],
    Difficulty.PROFESSIONAL: [200, 500],
}

AUDIENCES = {
    Difficulty.CHILD: "kindergarten students",
    Difficulty.PRIMARY: "primary school students",
    Difficulty.UNIVERSITY: "university students",
    Difficulty.PROFESSIONAL: "professionals",
}
