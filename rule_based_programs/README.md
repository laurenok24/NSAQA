Run all code from the root.

aqa_metaProgram.py:
Extract information from a dive clip necessary for scoring. Extracted data will be saved as a pickle file.
```
python rule_based_programs/aqa_metaProgram.py path/to/video.mp4
```

aqa_metaProgram_finediving.py:
Extract information from dive frames in the FineDiving dataset found at: https://github.com/xujinglin/FineDiving/tree/main.
Each dive in this dataset has an instance id (x, y).
```
python rule_based_programs/aqa_metaProgram_finediving.py x y

# e.g. for instance id ('01', 1) 
python rule_based_programs/aqa_metaProgram_finediving.py 01 1
```
