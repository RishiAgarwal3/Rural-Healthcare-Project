import json
import os

input_file = 'data/nlp/translated/translated.txt'
output_file = 'data/nlp/translated/calls.jsonl'

lines = open(input_file).read().strip().split('\n')
with open(output_file, 'w') as f:
    current_call = None
    current_text = []
    
    for line in lines:
        if 'Call ID: ' in line:
            if current_call:
                f.write(json.dumps({
                    'call_id': current_call,
                    'segments': [{'text_en': ' '.join(current_text)}]
                }) + '\n')
            current_call = line.split('Call ID: ')[1].strip()
            current_text = []
        else:
            if line.strip():
                current_text.append(line.strip())
                
    if current_call:
        f.write(json.dumps({
            'call_id': current_call,
            'segments': [{'text_en': ' '.join(current_text)}]
        }) + '\n')
