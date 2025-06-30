from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

import datetime

# Get the current date
current_date = datetime.date.today()

REFLECTION_PROMPT = " "
MAIN_PROMPT = " "
CONTEXTUALIZER_SYSTEM_PROMPT = " "
