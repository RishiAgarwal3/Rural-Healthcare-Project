_Excel Sheet to workable Audio files-_
step 1:use build_manifest.py 
Step 2 : use download_audio.py to download audio files from excel sheet
step 3 : use normalize_audio.py to normalise the audio into workable files

_Transciption To Charts_
step 1: use improved_transscription.py to transcribe the bengali text
step 2 : paste the transcribed text into an IDE tell it to joing the segments and translated contextually
step 3 : paste the results into translated.txt to generate calls.jsonl
step 4 : use parse_translated.txt to parse the text
step 5: to extract structured Healthcare Data - use -> "python -m src.nlp.format_enriched --export-csv"
step 6: to generate charts and analytics use -> "python scripts/eda/run_eda.py" remember to delete old results before running this


to run step 4,5,6 using 1 command - "python parse_translated.py && python -m src.nlp.format_enriched --export-csv && python scripts/eda/run_eda.py"


prompt for Ai 
can you join the segments and translate to english - dont hallucinate, try to keep the translations in context and dont add stuff that isnt directly mentioned - example-
