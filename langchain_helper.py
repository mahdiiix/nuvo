from langchain.tools import BaseTool, Tool
from langchain import HuggingFaceHub, LLMChain
from langchain.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain import hub
from typing import Type

class ColumnHookTool(BaseTool):
    name = "column_hook"
    description = '''Within this function, you have the ability to retrieve all the data from a 
        specific column. You can implement data modifications and present customized UI notifications, 
        including info messages, warnings, and/or errors.
        This tool returns a piece of javascript code to restructure phone numbers.'''

    def _run(self, columnName: str) -> str:
        return '''columnHooks={{{{
                    {columnName}: (values) => {{
                      values.map(([item, index]) => {{
                        let {columnName} = item;
                        if (/^[0]{{2}}/.test({columnName})) {{
                          {columnName} = `+${{item.slice(2, item.length)}}`;
                        }}
                        return [
                          {{
                            value: {columnName},
                            info: [
                              {{
                                message:
                                  'We have automatically transformed your input into the correct format by converting "00" to "+".',
                                level: "info",
                              }},
                            ],
                          }},
                          index,
                        ];
                      }});
                    }},
                  }}}}'''.format(columnName=columnName)
    
    def _arun(self, columnName: str) -> str:
        raise NotImplementedError("This tool does not support async")

tools = [ColumnHookTool()]
llm = Ollama(model="llama2", temperature = 0)
prompt = hub.pull("hwchase17/react")

sys_prompt = '''Answer the following questions as best you can. You are not good at modifying phone numbers so
for modifying or correcting a phone number just use the tools.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: You should always think about what to do.
Action: the action to take, should be one of ["{tool_names}"]. 
Action Input: the input to the action. The parameter is a string which is just a column name which should be
mentioned in the Question.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N-1 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. 

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt.template = sys_prompt

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

if __name__ == '__main__':
    result = agent_executor.invoke({'input':'I have a data frame with a phone number column with the name of "phones". phone numbers in this column are not in a correct format and I want to have javascript code to discard leading zeros and replace them with plus sign.'})
