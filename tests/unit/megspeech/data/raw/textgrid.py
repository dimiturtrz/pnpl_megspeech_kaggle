from pathlib import Path

from megspeech.data.raw.textgrid import TextGrid

GRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.0
tiers? <exists>
size = 2
item []:
	item [1]:
		class = "IntervalTier"
		name = "words"
		xmin = 0.0
		xmax = 1.0
		intervals: size = 2
			intervals [1]:
				xmin = 0.000
				xmax = 0.250
				text = "if"
			intervals [2]:
				xmin = 0.250
				xmax = 1.0
				text = ""
	item [2]:
		class = "IntervalTier"
		name = "phones"
		xmin = 0.0
		xmax = 1.0
		intervals: size = 1
			intervals [1]:
				xmin = 0.000
				xmax = 0.050
				text = "IH1"
"""


def grid(tmp_path: Path) -> TextGrid:
    path = tmp_path / "1.TextGrid"
    path.write_text(GRID, encoding="utf-8")
    return TextGrid.read(path)


def test_read(tmp_path: Path) -> None:
    read = grid(tmp_path)
    assert list(read.tiers) == ["words", "phones"]
    assert read.tiers["words"].to_dict("list") == {"start": [0.0, 0.25], "end": [0.25, 1.0], "text": ["if", ""]}
    assert read.tiers["phones"]["text"].tolist() == ["IH1"]


def test_aligned(tmp_path: Path) -> None:
    read = grid(tmp_path)
    aligned = TextGrid.aligned(read.phones, read.words)
    assert list(aligned.tiers) == ["words", "phones"]
    assert aligned.words is read.phones


def test_words(tmp_path: Path) -> None:
    assert grid(tmp_path).words["text"].tolist() == ["if", ""]


def test_phones(tmp_path: Path) -> None:
    assert grid(tmp_path).phones["text"].tolist() == ["IH1"]
