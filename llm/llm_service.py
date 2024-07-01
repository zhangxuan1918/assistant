from dataclasses import dataclass, field
import re
import threading
from typing import Tuple
from langchain_community.chat_models import ChatOllama
from prompt_util import ASSISTANT_PROMPT, SYSTEM_PROMPT, SYSTEM_ROLE, USER_INPUT, USER_PROMPT
from llm_manager import LlmManager, LlmGenerationTask, LlmGenerationResult, TaskStatus
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SENTENCE_END_PATTERN = r'[A-Za-z]+[\.\?\!]$'

@dataclass
class LLMService:
    llm_manager: LlmManager
    ollama_base_url: str = field(default="http://192.168.1.26:11434")
    model_name: str = field(default="llama3")
    model_temparature: float = field(default=0.7)
    stop_event: threading.Event = field(default_factory=threading.Event)
    system_prompt: str = field(default=SYSTEM_ROLE)
    stream_min_num_tokens_to_emit: int = field(default=30)

    def __post_init__(self):
        self.llm = ChatOllama(model=self.model_name, temperature=self.model_temparature, base_url=self.ollama_base_url)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.format(prompt=self.system_prompt)),
                ("user", USER_PROMPT.format(prompt=USER_INPUT)),
                ("assistant", ASSISTANT_PROMPT),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run(self, streaming=False):
        while not self.stop_event.is_set():
            if self.llm_manager.has_pending_text_gen_tasks():
                task = self.llm_manager.get_text_gen_task()
                if task is not None:
                    self.llm_manager.set_task_status(task_id=task.task_id, status=TaskStatus.RUNNING)
                    for response in self.convert(task):
                        self.llm_manager.save_text_gen_task(
                            LlmGenerationResult(task=task, response=response)
                        )
                    self.llm_manager.set_task_status(task_id=task.task_id, status=TaskStatus.FINISHED)

    def convert(self, task: LlmGenerationTask):
        text = ""
        num_tokens = 0
        for chunk in self.chain.stream({"context": task.context, "question": task.question}):
            text += chunk
            num_tokens += 1
            if self._should_emit(text, num_tokens):
                yield text
                text = ""
                num_tokens = 0
        if text:
            yield text

    # def _process_chunk(self, chunks: Array[str]) -> str:
    #     return ''.join(chunks)

    def _should_emit(self, text: str, num_tokens: int) -> bool:
        if num_tokens >= self.stream_min_num_tokens_to_emit and self._is_end_of_sentence(text):
            return True
        return False
    
    def _is_end_of_sentence(self, text: str) -> bool:
        return bool(re.search(SENTENCE_END_PATTERN, text))
    
    def stop(self):
        self.stop_event.set()
    
def start_llm(llm_manager: LlmManager) -> Tuple[LLMService, threading.Thread]:
    llm_service = LLMService(llm_manager)
    thread = threading.Thread(target=llm_service.run)
    thread.start()
    return llm_service, thread


def stop_llm(llm_service: LLMService, thread: threading.Thread) -> None:
    llm_service.stop()
    thread.join()


if __name__ == "__main__":
    import time

    context = """LangChain is a framework designed to facilitate the development of applications that leverage large language models (LLMs). It is particularly useful for building complex and dynamic applications that need to integrate language models with other computational and data-handling processes. LangChain provides a suite of tools and abstractions to streamline the development of such applications, enabling developers to harness the power of LLMs in a more structured and scalable way.

### Key Features of LangChain

1. **Chaining**:
   - LangChain allows you to create sequences (or chains) of operations where the output of one step serves as the input for the next. This is particularly useful for applications that need to perform multi-step reasoning or processing.

2. **Integration**:
   - It provides integration with various data sources and APIs, enabling LLMs to interact with external databases, web services, and more. This makes it easier to build applications that require real-time data retrieval or interaction with other software systems.

3. **Customization**:
   - LangChain offers a high level of customization, allowing developers to define custom components and logic for their specific use cases. This includes the ability to create custom prompts, handle specific types of user interactions, and implement unique processing steps.

4. **Efficiency**:
   - By organizing and managing the interactions with LLMs in a structured way, LangChain helps improve the efficiency and reliability of the applications. It helps manage the complexity that comes with handling stateful interactions and multi-step processes.

5. **Modularity**:
   - The framework is designed to be modular, making it easy to plug in different components and services as needed. This modularity supports the reuse of components and simplifies the process of updating and maintaining applications.

### Typical Use Cases

LangChain is suitable for a variety of applications, including but not limited to:

- **Chatbots and Virtual Assistants**: Building sophisticated conversational agents that can handle multi-turn dialogues and integrate with backend systems for fetching data or performing actions.
- **Content Generation**: Automating the creation of articles, reports, or other written content by chaining different generation and editing steps.
- **Data Analysis**: Using LLMs to interpret and analyze large datasets by integrating with data processing tools and databases.
- **Task Automation**: Creating workflows that automate complex tasks involving multiple steps and interactions with various services.

### Example Workflow

A simple example of how LangChain might be used in a chatbot application could involve the following steps:

1. **User Input Handling**:
   - Capture the user's query.
   
2. **Intent Recognition**:
   - Use an LLM to determine the user's intent.
   
3. **Data Retrieval**:
   - Fetch necessary data from an external API or database based on the recognized intent.
   
4. **Response Generation**:
   - Use the LLM to generate a response based on the user's query and the retrieved data.
   
5. **Output Delivery**:
   - Present the generated response to the user.

### Example Code Snippet

Here is a hypothetical example to illustrate how LangChain might be used in Python:

```python
from langchain import Chain, LLM, DataConnector

# Define a language model
llm = LLM(model_name="gpt-4")

# Define a data connector to fetch weather data
weather_data_connector = DataConnector(api_url="https://api.weather.com/v3/wx/conditions/current")

# Define the processing chain
class WeatherChain(Chain):
    def __init__(self, llm, data_connector):
        self.llm = llm
        self.data_connector = data_connector
    
    def run(self, user_input):
        # Step 1: Recognize intent (e.g., checking weather)
        intent = self.llm.predict(f"Determine intent: {user_input}")
        
        if "weather" in intent:
            # Step 2: Fetch weather data
            location = self.llm.predict(f"Extract location: {user_input}")
            weather_data = self.data_connector.get_data(params={"location": location})
            
            # Step 3: Generate response
            response = self.llm.predict(f"Generate weather response: {weather_data}")
            return response
        else:
            return "I'm not sure how to help with that."

# Create an instance of the chain
weather_chain = WeatherChain(llm, weather_data_connector)

# Run the chain with user input
user_query = "What's the weather like in New York?"
response = weather_chain.run(user_query)
print(response)
```

In this example, the `WeatherChain` class defines a sequence of steps to handle a user's query about the weather, integrating LLM predictions and external data fetching seamlessly.

### Conclusion

LangChain provides a powerful framework for building sophisticated applications that leverage the capabilities of large language models. By offering tools for chaining operations, integrating with external systems, and customizing workflows, LangChain simplifies the development process and helps create more efficient and effective language model-based applications."""
    question1 = """I'm working on building an assistant using LLM. How should I use langchain?"""
    question2 = """In what cases, should I avoid using langchain?"""
    user_inputs = [(context, question1), (context, question2)]
    
    llm_manager = LlmManager(conversation_id="1")
    llm_service, thread = start_llm(llm_manager=llm_manager)

    for i, (context, question) in enumerate(user_inputs):
        print(f"processing {i}th input: {question}")
        task = LlmGenerationTask(task_id=str(i), context=context, question=question)
        llm_manager.add_text_gen_task(task)

        index, response = 0, ""
        task_status = TaskStatus.UNKNOWN
        while (task_status := llm_manager.get_task_status(task_id=task.task_id)) and task_status != TaskStatus.UNKNOWN:
            response = llm_manager.get_text_gen_result(task_id=task.task_id, index=index)
            if response is not None:
                index += 1
                print(response, end="")
            elif task_status == TaskStatus.FINISHED:
                print("\n")
                break
        print("----------------------------------------------------------------------------------")

    stop_llm(llm_service=llm_service, thread=thread)
