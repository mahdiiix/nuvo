from fastapi import FastAPI
from langchain_helper import agent_executor

app = FastAPI()

@app.get("/clearphone")
def clear_phone(column_name: str = 'phones'):
    query = f'I have a data frame with a phone number column with the name of "{column_name}". phone numbers in this column are not in a correct format and I want to have javascript code to discard leading zeros and replace them with plus sign.'
    result = agent_executor.invoke({'input': query})
    return result
