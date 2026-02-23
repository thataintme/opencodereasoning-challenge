import pandas as pd

df = pd.read_csv('easy_python_questions.csv')
for index, row in df.iterrows():
    question_id = row['question_id']
    id = row['id']
    judgement = row['judgement']

print()

# Discontinued, cuz at this point i discovered that questions are duplicated