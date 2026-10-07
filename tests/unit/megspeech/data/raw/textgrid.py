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


def test_read(tmp_path: Path) -> None:
    path = tmp_path / "1.TextGrid"
    path.write_text(GRID, encoding="utf-8")
    grid = TextGrid.read(path)
    assert list(grid.tiers) == ["words", "phones"]
    assert grid.tiers["words"].to_dict("list") == {"start": [0.0, 0.25], "end": [0.25, 1.0], "text": ["if", ""]}
    assert grid.tiers["phones"]["text"].tolist() == ["IH1"]
