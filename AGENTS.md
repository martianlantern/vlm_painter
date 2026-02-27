I prefer clean, minimal and less cluttered code with no comments. The code should be as short as possible and you should only write code that is needed. There should be no comments in the code. Writing short code does not mean it should not handle complex code logic, it absolutely can and should handle complex behaviours if the task demands it to, it should be able to function and perform robustly and not be fragile. But the code should be short

When creating python script prefer making the scripts an executable using `chmod +x script_name.py` and running it with `./script_name.py` using uv run --script and inline python dependencies like the following
```python
#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fire>=0.7.1",
#     ...
# ]
# ///
def main():
    pass
if __name__ == "__main__":
    fire.Fire(main)
```
always use the fire library for creating and passing args in the script when ever needed
If the task does not allow create the python executable and inline dependencies it's completely fine to prefer an alternative instead

After every experiment/implementation or sprint maintain a note of what you did under `notes` dir. The note can include things like what was the goal, what was the outcome, how you did it, etc, Also feel free to log any results or numbers or even images (via ![]() in md) or media assets in the notes if they are important or seems like it needs to be showed to understand the context of the note and the project

The notes are for yourself and not for anyone else, think of them like your rough notebook or scratchpad where you can store or offload your work for later use, this will help you to later remember and re read what you did, what worked and how you did it. The idea with this is that even if you lose your context or switch to a new session, you can still get some context from previous sessions also via these notes and also better plan everything ahead. This is how I want to structure the notes

```markdown
---
title: short_title_for_the_note
description: summary_of_the_entire_note
created_at: 3:40PM, 27th Feb, 2026
finished_at: 11:40PM, 28th Feb, 2026 (or None)
pending: True/False
ongoing: True/False
finished: True/False
related notes: [[note_1.md]], [[note_2.md]], [[note_3.md]]
---

description of the note
contents of the notes
logs, summaries, results, ablations or anything else
outcome etc
...
...
anything that you want to store here for future use

```
The related notes will help to create an obsidian like graph of the notes and how the knowledge and the project is evolving over time, this will also be helpful so also maintain this related notes properties

If you are just starting to work on a new project, it's always helpful to read the latest notes from the `notes` dir to get context about the project, if no notes are present that means we have just started and should ensure that `notes` dir is created