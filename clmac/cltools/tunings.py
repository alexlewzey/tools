"""Script that allows you to work out lower scale tunings on a 7-string
guitar."""

notes = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

tunings = {"standard": ["B", "E", "A", "D", "G", "B", "E"]}


def make_string(note):
    return list(reversed(notes + notes[: notes.index(note) + 1]))


def lower_tuning(semitones: int):
    return [make_string(note)[semitones] for note in tunings["standard"]]


def show_tuning(semitones):
    print(lower_tuning(semitones))
