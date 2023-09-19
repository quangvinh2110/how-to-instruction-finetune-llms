def make_slot_prompt(instruction, input):
    prompt = (
f"""### Instruction: 
{instruction}

### Input:
{input}

### Response:"""
)
    return prompt
# END generate_qna_prompt

# import json
# import pandas as pd
# dst = pd.read_csv("data/dst_training_data.csv")
# with open("data/translated_slots.jsonl", "w") as f:
#      for index, row in dst.iterrows():
#          text = row['text']
#          target = str( row['target'])
#          if not target: target = "NONE"
#          components = text.split("<sep>")
#          input = components[0]
#          instruction = " ".join(components[1:3])
#          response = components[3] + " " + target
#          d = json.dumps({"instruction": instruction, "input": input, "response": response})
#          f.write(d + "\n")
