import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from yutalk.utils import setup_logger

logger = setup_logger(name="yutalk_graph", level=logging.INFO, log_file="yutalk.log")


MODEL_NAME = "gpt-4o-mini"

# System prompts
GRAMMAR_CHECK_PROMPT = """
You are a Chinese language learning assistant specialized in grammar correction.
Analyze the following Chinese text for grammatical errors, unnatural phrasing, or word choice issues.

For each issue found:
1. Identify the specific error
2. Explain why it's incorrect
3. Provide the correct version
4. Give a brief explanation that would help a language learner understand

Example:
Human: 我每天吃饭三次。
You: 句子“我每天吃饭三次”语序错误。正确应为：“我每天吃三次饭。” 原因：表示动作频率时，结构应为：主语 + 时间 + 动词 + 次数 + 宾语。

If the text is grammatically correct and natural sounding, simply respond with "CORRECT: The text is grammatically correct and sounds natural."

Remember that your goal is to help a language learner improve their Chinese, so be thorough but encouraging.
"""

CONVERSATION_PROMPT = """
You are a friendly Chinese conversation partner helping a language learner practice Chinese.
Your role is to maintain a natural, engaging conversation while being aware of the language learner's level.

Guidelines:
- Respond naturally to the user's message in Chinese
- Use vocabulary and grammar structures appropriate for their level
- Keep responses concise (2-3 sentences) to maintain conversation flow
- Ask open-ended questions to encourage more practice
- Be patient and encouraging

The conversation should feel natural, like chatting with a friend, while still providing gentle learning opportunities.
"""

class GrammarCheckResult(BaseModel):
    is_correct: bool = Field(description="Whether the text is grammatically correct")
    corrections: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="List of corrections with original text, corrected text, and explanations"
    )
    original_text: str = Field(description="The original text that was checked")

class Transcript(BaseModel):
    original: str = Field(description="Original transcribed text from the user")
    corrected: Optional[str] = Field(default=None, description="Corrected version if grammar issues were found")
    grammar_feedback: Optional[GrammarCheckResult] = Field(default=None, description="Grammar check results")

class UserPreferences(BaseModel):
    correction_level: str = Field(
        default="balanced", 
        description="How detailed corrections should be: 'minimal', 'balanced', or 'comprehensive'"
    )
    topics_of_interest: List[str] = Field(
        default_factory=lambda: ["daily life", "travel", "food"], 
        description="Topics the user is interested in discussing"
    )
    skill_level: str = Field(
        default="intermediate", 
        description="User's Chinese proficiency level: 'beginner', 'intermediate', or 'advanced'"
    )


class State(TypedDict):
    """State definition for the YuTalk conversation graph."""
    messages: Annotated[list, add_messages]
    transcripts: List[Transcript]
    user_preferences: UserPreferences
    current_topic: Optional[str]
    session_id: str


class YuTalkGraph:
    def __init__(self, user_preferences: Optional[UserPreferences] = None, session_id: str = "default"):
        """
        Initialize a YuTalk conversation graph.

        Args:
            user_preferences (UserPreferences, optional): User preferences for corrections and conversation
            session_id (str): Unique identifier for this conversation session
        """
        self.session_id = session_id
        self.user_preferences = user_preferences or UserPreferences()
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
        self.grammar_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)  # Lower temperature for grammar checking
        self.config = {"configurable": {"thread_id": f"yutalk_{session_id}"}}
        self.graph = self._build_graph()

    def _grammar_check_node(self, state: State) -> Dict:
        """
        Check grammar of the transcribed Chinese text.
        """
        messages = state["messages"]
        
        # Get the last message which should be the transcribed text
        if not messages or not isinstance(messages[-1], HumanMessage):
            raise ValueError("Expected a human message with transcribed text")
        
        transcribed_text = messages[-1].content
        
        # Create a system message for grammar checking
        system_msg = SystemMessage(content=GRAMMAR_CHECK_PROMPT)
        
        # Create a prompt for grammar checking
        check_prompt = HumanMessage(content=transcribed_text)
        
        try:
            # Invoke the LLM for grammar checking
            response = self.grammar_llm.invoke([system_msg, check_prompt])
            
            # Parse the response to determine if text is correct or needs corrections
            response_text = response.content
            
            if response_text.startswith("CORRECT:"):
                grammar_result = GrammarCheckResult(
                    is_correct=True,
                    corrections=None,
                    original_text=transcribed_text
                )
            else:
                # Extract structured feedback from the response
                # This is a simplified parsing - in production you might want more robust parsing
                grammar_result = GrammarCheckResult(
                    is_correct=False,
                    corrections=[{"explanation": response_text}],  # Simplified for this example
                    original_text=transcribed_text
                )
            
            # Create transcript record
            transcript = Transcript(
                original=transcribed_text,
                grammar_feedback=grammar_result,
                corrected=None  # Will be populated if we implement correction generation
            )
            
            # Add to state
            state["transcripts"].append(transcript)
            
            return {"transcripts": state["transcripts"]}
            
        except Exception as e:
            logger.error(f"Error in grammar checking: {str(e)}")
            raise Exception(f"Error checking grammar: {str(e)}")

    def _respond_to_user_node(self, state: State) -> Dict:
        """
        Generate a conversational response to the user's message.
        """
        messages = state["messages"]
        user_preferences = state["user_preferences"]
        
        # Create context with conversation history
        conversation_history = []
        
        # Add system message with conversation instructions
        system_prompt = CONVERSATION_PROMPT
        
        # Add user preference information to the system prompt
        system_prompt += f"\n\nUser preferences:"
        system_prompt += f"\n- Skill level: {user_preferences.skill_level}"
        system_prompt += f"\n- Correction level: {user_preferences.correction_level}"
        system_prompt += f"\n- Topics of interest: {', '.join(user_preferences.topics_of_interest)}"
        
        system_msg = SystemMessage(content=system_prompt)
        conversation_history.append(system_msg)
        
        # Add relevant conversation history
        # In a real implementation, you might want to limit this to recent messages
        for msg in messages:
            conversation_history.append(msg)
        
        try:
            # Generate response
            response = self.llm.invoke(conversation_history)
            
            # Return updated state with the AI response added
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise Exception(f"Error generating response: {str(e)}")

    def _build_graph(self) -> StateGraph:
        """
        Build the graph structure.
        """
        graph_builder = StateGraph(State)

        # Add nodes to the graph
        graph_builder.add_node("grammar_check", self._grammar_check_node)
        graph_builder.add_node("respond_to_user", self._respond_to_user_node)
        
        # Define the edges in the graph
        graph_builder.add_edge("grammar_check", "respond_to_user")
        
        # Set the entry point
        graph_builder.set_entry_point("grammar_check")
        
        return graph_builder.compile(checkpointer=MemorySaver())

    async def process_user_input_stream(self, transcribed_text: str) -> AsyncIterator[Dict]:
        """
        Process user input and generate a response with streaming support.

        Args:
            transcribed_text (str): The transcribed Chinese text from Whisper

        Returns:
            AsyncIterator[Dict]: Stream of updates from the processing
        """
        try:
            # Create the initial state
            initial_state = {
                "messages": [HumanMessage(content=transcribed_text)],
                "transcripts": [],
                "user_preferences": self.user_preferences,
                "current_topic": None,
                "session_id": self.session_id
            }

            # Stream both values and messages
            async for chunk in self.graph.astream(
                initial_state, self.config, stream_mode=["values", "messages"]
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            raise Exception(f"Error processing user input: {str(e)}")


# Factory function to get or create YuTalk graphs
_yutalk_graphs = {}

def get_yutalk_graph(
    session_id: str, 
    user_preferences: Optional[UserPreferences] = None
) -> YuTalkGraph:
    """
    Get or create a YuTalk conversation graph.

    Args:
        session_id (str): Unique identifier for this conversation session
        user_preferences (UserPreferences, optional): User preferences for corrections and conversation

    Returns:
        YuTalkGraph: The YuTalk graph instance
    """
    if session_id not in _yutalk_graphs:
        _yutalk_graphs[session_id] = YuTalkGraph(user_preferences, session_id)
    return _yutalk_graphs[session_id]