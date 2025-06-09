from dataclasses import dataclass


@dataclass
class Tokens:
    input: int
    output: int

    def __add__(self, other):
        return Tokens(
            input=self.input + other.input,
            output=self.output + other.output,
        )
