"""Uv run --only-group guitartk python -m src.guitartk.guitartk."""

import numpy as np
from rich.console import Console

np.set_printoptions(threshold=None, linewidth=200, edgeitems=None)


console = Console()
chromatic_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

scale_intervals: dict[str, list[int]] = {
    "chromatic": list(range(1, 12)),
    "major": [2, 2, 1, 2, 2, 2, 1],
    "minor": [2, 1, 2, 2, 1, 2, 2],
    "lonian": [2, 2, 1, 2, 2, 2, 1],
    "Dorian": [2, 1, 2, 2, 2, 1, 2],
    "Phrygian": [1, 2, 2, 2, 1, 2, 2],
    "Lydian": [2, 2, 2, 1, 2, 2, 1],
    "Mixolydian": [2, 2, 1, 2, 2, 1, 2],
    "Aeolian": [2, 1, 2, 2, 1, 2, 2],
    "Locrian": [1, 2, 2, 1, 2, 2, 2],
}


def count_semitones(note1, note2):
    i1 = chromatic_notes.index(note1)
    i2 = chromatic_notes.index(note2)
    return (i2 - i1) % 12


def add_semitones(note1, n_semitones):
    i1 = chromatic_notes.index(note1)
    i2 = (i1 + n_semitones) % 12
    return chromatic_notes[i2]


def tuning_to_intervals(tuning_str: str):
    tuning = tuning_str.split()
    intervals = []
    for i in range(len(tuning) - 1):
        intervals.append(count_semitones(tuning[i], tuning[i + 1]))
    print(f"'{tuning_str}', ", intervals)


def intervals_to_tuning(intervals, start_note):
    tuning = [start_note]
    note = start_note
    for interval in intervals:
        note = add_semitones(note, interval)
        tuning.append(note)
    return tuning


db = [
    ("northlane", "F A# F A# D# G C", [5, 7, 5, 5, 4, 5]),
    ("fata morgana", "B A D G C F A D", [10, 5, 5, 5, 5]),
    ("e standard", "E A D G B E", [5, 5, 5, 4, 5]),
    ("spirit box", "F# C# F# B E G# C#", [7, 5, 5, 5, 4, 5]),
    ("thornhill", "F# C# F# B E G# C#", [7, 5, 5, 5, 4, 5]),
    ("meshuggah", "F A# D# G# C# F A#", [5, 5, 5, 5, 4, 5]),
]

tuning_str = "B A D G C F A D"
tuning_to_intervals(tuning_str)
tuning_str = "D A D G B E"
tuning_to_intervals(tuning_str)
intervals_to_tuning([10, 5, 5, 5, 5], "F")
notes_scale = intervals_to_tuning(scale_intervals["minor"], "F")


def to_color(s: str) -> str:
    return f"[red]{s}[/red]"


def print_neck(note, intervals, color_notes: list[str] = None):
    frets = []
    for i in range(25):
        fret = intervals_to_tuning(intervals, add_semitones(note, i))
        frets.append(fret[::-1])
    neck = np.char.ljust(np.array(frets).T, 2)
    out: str = ""
    for idx, row in enumerate(neck):
        if color_notes:
            row = [
                to_color(note) if note.strip() in color_notes else note for note in row
            ]
        out += "|".join([str(len(neck) - (idx)) + " "] + list(row)) + "\n"

    positions = [0, 3, 5, 7, 10, 12, 15, 17, 21, 24]
    marker_row = []
    for i in range(25):
        if i in positions:
            marker_row.append(str(i).ljust(2))
        else:
            marker_row.append("  ")
    out += "|".join(["  "] + marker_row)
    console.print(out)


note = "F"
intervals = [10, 5, 5, 5, 5]
console.print("fata morgana")
print_neck(note, intervals, notes_scale)
note = "F"
intervals = [5, 7, 5, 5, 4, 5]
console.print("schecter")
print_neck(note, intervals, notes_scale)
