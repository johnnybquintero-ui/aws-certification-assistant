from src.simple_interface import run_interface
from tests.conftest import FakeClassifier

def test_interface_displays_classification(monkeypatch, capsys):
    # Turn the test responses into an iterator.
    # Each call to next(responses) returns the next value:
    # first the description, then the exit command.
    responses = iter(
        [
            "I need somewhere to store objects",
            "exit",
        ]
    )

    # Temporarily replace Python's real input() function.
    #
    # When run_interface() calls input(), it will receive the next
    # value from responses instead of waiting for keyboard input.
    #
    # The `prompt` argument receives the prompt passed to input(),
    # but this fake function does not need to use it.
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: next(responses),
    )

    classifier = FakeClassifier()

    # Run the real interface using the fake classifier and fake input().
    run_interface(classifier)

    # capsys has captured everything printed by run_interface().
    #
    # readouterr() returns:
    # - .out for normal printed output
    # - .err for error output
    output = capsys.readouterr().out

    # Check that the interface sent the user's description
    # to the classifier.
    assert classifier.received_inputs == [
        "I need somewhere to store objects"
    ]

    # Check that the interface displayed the classifier's result.
    assert "[Result] s3" in output

    # Check that the exit command ended the interface correctly.
    assert "Goodbye!" in output