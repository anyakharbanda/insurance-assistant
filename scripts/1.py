import google.generativeai as genai
genai.configure(api_key="AIzaSyCz1cSWFSMOFibAy9Yb7Dqq5SFZHEMZZZ0")

for m in genai.list_models():
    print(m.name)
