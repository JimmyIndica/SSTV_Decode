# Working on robot36-decode with git

## One-time setup

    git config --global user.name  "Your Name"
    git config --global user.email "you@example.com"

## Get the code

    git clone <url-of-this-repository>
    cd robot36-decode-repo

## Run it and the tests

    py -m pip install numpy scipy                 (Windows)
    python3 -m pip install numpy scipy            (macOS / Linux)
    py -m robot36_decode capture.wav              (decode a file)
    py tests/test_roundtrip.py                    (all tests, ~2 minutes, must end "N/N tests passed")

## Making a change (the whole workflow)

    git switch -c my-change        # 1. new branch; never work on main directly
    ... edit files, run the tests ...
    git status                     # 2. what did I touch?
    git diff                       #    and exactly how?
    git add -A                     # 3. stage it
    git commit -m "Short summary of what and why"
    git push -u origin my-change   # 4. publish the branch, then open a merge/pull request

## Releasing

1. Bump `__version__` in `robot36_decode/cli.py`, add a line to the Version history in `README.md`
   and update the test count there.
2. Run the tests.
3. Commit, merge to `main`, then tag: `git tag -a v3.5.0 -m "v3.5.0"` and `git push --tags`.

## Rules of thumb

* Fixes should be DSP-level (how the signal is measured), not cosmetic post-processing of the picture.
* Do not change spec timing (pixel offsets) without evidence from real captures.
* New behaviour gets a test in `tests/test_roundtrip.py`; the file's runner (`_run_all`) stays at the end.
* Do not commit captures or decoded PNGs (they are git-ignored). The tests build their own synthetic signals and delete their temporary files when they finish (`R36_KEEP_TMP=1` keeps them).
