import traceback


def invariant(clause: bool):
    """
    Used to assert invariants. If assertion is false, then it outputs the error to terminal, but does not terminate execution.
    """
    try:
        assert clause
    except AssertionError as e:
        traceback.print_exception(e)
