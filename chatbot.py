
import re
import random
import json
import datetime
import os
from collections import defaultdict

class ConversationalAIBot:
    def __init__(self, name="ConvoBot"):
        self.name = name
        self.user_name = "User"
        self.user_preferences = {}
        self.conversation_history = []
        self.session_counter = 0
        self.last_interaction_time = None
        self.topic_history = []
        self.current_topic = None
        self.emotion_detected = None
        self.system_prompt = self._load_prompts("system")
        self.topic_prompts = {
            "information": self._load_prompts("information"),
            "recommendation": self._load_prompts("recommendation"),
            "problem_solving": self._load_prompts("problem_solving"),
            "emotional_support": self._load_prompts("emotional_support")
        }
        self.context_prompts = {
            "initial": self._load_prompts("initial"),
            "returning": self._load_prompts("returning"),
            "transition": self._load_prompts("transition")
        }
        self.response_templates = {
            "information": self._load_templates("information"),
            "recommendation": self._load_templates("recommendation"),
            "problem_solving": self._load_templates("problem_solving"),
            "emotional_support": self._load_templates("emotional_support"),
            "greeting": self._load_templates("greeting"),
            "farewell": self._load_templates("farewell")
        }
        
        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Load user profile if exists
        self.user_profile = self._load_user_profile()
        
        # Initialize NLP tools
        self.intents = self._initialize_intents()
        self.entities = {}
        
    def _load_prompts(self, prompt_type):
        """Load prompts from configuration or use defaults."""
        default_prompts = {
            "system": "You are ConvoBot, a helpful and engaging conversational AI assistant designed to have natural, meaningful conversations with users.",
            "information": "Provide accurate and clear information on the requested topic.",
            "recommendation": "Offer personalized recommendations based on user preferences.",
            "problem_solving": "Help users solve problems by breaking them down into manageable steps.",
            "emotional_support": "Provide empathetic support and understanding.",
            "initial": "Welcome new users warmly and learn about their interests.",
            "returning": "Remember details from previous conversations with returning users.",
            "transition": "Handle topic transitions smoothly and naturally."
        }
        
        # In a real implementation, you might load these from a file
        return default_prompts.get(prompt_type, default_prompts["system"])
    
    def _load_templates(self, template_type):
        """Load response templates."""
        templates = {
            "information": [
                "{direct_answer}\n\n{supporting_details}\n\n{multiple_perspectives}\n\n{knowledge_limitations}\n\n{related_topics}\n\nIs there anything specific about {topic} you'd like to explore further?"
            ],
            "recommendation": [
                "Based on {preferences}, I'd recommend considering {primary_recommendation}.\n\n{explanation}\n\nSome other options you might consider:\n- {alt1}: {alt1_explanation}\n- {alt2}: {alt2_explanation}\n- {alt3}: {alt3_explanation}\n\nWhat aspects of these recommendations sound most appealing to you?"
            ],
            "problem_solving": [
                "I understand you're facing {problem}.\n\nHere are some approaches you might consider:\n\n1. {solution1}\n   - {solution1_details}\n   - {solution1_pros_cons}\n\n2. {solution2}\n   - {solution2_details}\n   - {solution2_pros_cons}\n\nWould you like more details about any of these approaches?"
            ],
            "emotional_support": [
                "I can understand why you might feel {emotion} about {situation}.\n\n{validation}\n\n{perspective}\n\n{support_offer}\n\nWould it help to talk more about this, or would you prefer to focus on something else?"
            ],
            "greeting": [
                "Hello! I'm {bot_name}. How can I help you today?",
                "Hi there! I'm {bot_name}. What would you like to talk about?",
                "Welcome! I'm {bot_name}. How are you doing today?",
                "Greetings! I'm {bot_name}, your conversational assistant. What brings you here today?"
            ],
            "farewell": [
                "Goodbye! It was nice chatting with you.",
                "Until next time! Feel free to return whenever you'd like to chat again.",
                "Take care! I enjoyed our conversation.",
                "Farewell! I'll be here if you need assistance in the future."
            ]
        }
        
        return templates.get(template_type, ["I'm not sure how to respond to that."])
    
    def _initialize_knowledge_base(self):
        """Initialize the chatbot's knowledge base."""
        return {
            "general_facts": {
                "capital_cities": {
                    "USA": "Washington D.C.",
                    "UK": "London",
                    "France": "Paris",
                    "Japan": "Tokyo",
                    "Australia": "Canberra"
                },
                "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
            },
            "recommendations": {
                "books": {
                    "fiction": ["To Kill a Mockingbird", "1984", "The Great Gatsby", "Pride and Prejudice"],
                    "non_fiction": ["Sapiens", "Thinking, Fast and Slow", "Educated", "The Immortal Life of Henrietta Lacks"],
                    "science_fiction": ["Dune", "Foundation", "Neuromancer", "The Left Hand of Darkness"]
                },
                "movies": {
                    "action": ["The Matrix", "Die Hard", "Mad Max: Fury Road"],
                    "drama": ["The Shawshank Redemption", "The Godfather", "Schindler's List"],
                    "comedy": ["The Grand Budapest Hotel", "Airplane!", "Monty Python and the Holy Grail"]
                }
            },
            "functions": {
                "tell_time": lambda: f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}.",
                "tell_date": lambda: f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}.",
                "tell_joke": lambda: random.choice([
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "What do you call a fake noodle? An impasta!",
                    "Why did the Python programmer wear glasses? Because they couldn't C#!",
                    "How many programmers does it take to change a light bulb? None, that's a hardware problem!"
                ])
            }
        }
    
    def _initialize_intents(self):
        """Initialize intent recognition patterns."""
        return {
            "greeting": r"(?i)hello|hi|hey|greetings|good morning|good afternoon|good evening",
            "farewell": r"(?i)bye|goodbye|see you|farewell|exit|quit",
            "gratitude": r"(?i)thank you|thanks|appreciate|grateful",
            "information_request": r"(?i)what is|who is|where is|when is|why is|how does|can you tell me about|explain|describe",
            "recommendation_request": r"(?i)recommend|suggest|what should I|best|top|favorite|advice on",
            "problem_statement": r"(?i)I need help with|I'm having trouble with|I can't figure out|How do I solve|fix|resolve|issue|problem",
            "emotional_expression": r"(?i)I feel|I am feeling|I'm sad|I'm happy|I'm worried|I'm excited|I'm confused|I'm frustrated",
            "preference_statement": r"(?i)I like|I love|I prefer|I enjoy|I don't like|I hate|favorite",
            "clarification": r"(?i)what do you mean|I don't understand|could you explain|can you clarify",
            "identity_question": r"(?i)who are you|what are you|your name|about you",
            "capability_question": r"(?i)what can you do|help me with|your capabilities|can you",
            "time_request": r"(?i)what time is it|current time|the time",
            "date_request": r"(?i)what day is it|what is the date|today's date|current date",
            "joke_request": r"(?i)tell me a joke|know any jokes|something funny"
        }
    
    def _load_user_profile(self):
        """Load user profile from storage or create a new one."""
        try:
            with open("user_profile.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"preferences": {}, "topics_discussed": [], "session_count": 0}
    
    def _save_user_profile(self):
        """Save user profile to storage."""
        with open("user_profile.json", "w") as f:
            json.dump(self.user_profile, f)
    
    def _detect_intent(self, user_input):
        """Detect the user's intent based on their input."""
        for intent, pattern in self.intents.items():
            if re.search(pattern, user_input):
                return intent
        return "unknown"
    
    def _extract_entities(self, user_input):
        """Extract named entities from user input."""
        # In a real implementation, this would use NLP techniques
        # This is a simplified version for demonstration
        entities = {}
        
        # Extract user name
        name_match = re.search(r"(?i)my name is (\w+)", user_input)
        if name_match:
            entities["user_name"] = name_match.group(1)
        
        # Extract preferences
        preference_match = re.search(r"(?i)I (?:like|love|enjoy) (\w+)", user_input)
        if preference_match:
            entities["preference"] = preference_match.group(1)
        
        # Extract negative preferences
        dislike_match = re.search(r"(?i)I (?:don't like|hate|dislike) (\w+)", user_input)
        if dislike_match:
            entities["dislike"] = dislike_match.group(1)
        
        return entities
    
    def _detect_emotion(self, user_input):
        """Detect emotional tone in user input."""
        emotions = {
            "happy": r"(?i)happy|glad|excited|delighted|pleased|joy|wonderful|great",
            "sad": r"(?i)sad|unhappy|depressed|down|miserable|upset|heartbroken",
            "angry": r"(?i)angry|mad|frustrated|annoyed|irritated|furious",
            "confused": r"(?i)confused|unsure|uncertain|don't understand|unclear",
            "anxious": r"(?i)anxious|worried|nervous|stressed|concerned|afraid|scared"
        }
        
        for emotion, pattern in emotions.items():
            if re.search(pattern, user_input):
                return emotion
        
        return None
    
    def _detect_topic(self, user_input):
        """Detect the topic of the conversation."""
        topics = {
            "technology": r"(?i)computer|technology|software|hardware|programming|code|app|website|internet|smartphone",
            "entertainment": r"(?i)movie|film|show|book|music|game|entertainment|art|theater|concert",
            "food": r"(?i)food|restaurant|recipe|cook|eat|drink|meal|diet|nutrition",
            "travel": r"(?i)travel|vacation|trip|visit|country|city|flight|hotel|tourism",
            "health": r"(?i)health|fitness|exercise|diet|medical|doctor|illness|disease|wellness",
            "education": r"(?i)education|school|college|university|learn|study|teach|course|degree"
        }
        
        for topic, pattern in topics.items():
            if re.search(pattern, user_input):
                return topic
        
        return None
    
    def _update_context(self, user_input, intent, entities):
        """Update conversation context based on user input."""
        # Update last interaction time
        self.last_interaction_time = datetime.datetime.now()
        
        # Update user name if provided
        if "user_name" in entities:
            self.user_name = entities["user_name"]
            self.user_profile["name"] = entities["user_name"]
        
        # Update user preferences
        if "preference" in entities:
            self.user_preferences[entities["preference"]] = True
            if "preferences" not in self.user_profile:
                self.user_profile["preferences"] = {}
            self.user_profile["preferences"][entities["preference"]] = True
        
        if "dislike" in entities:
            self.user_preferences[entities["dislike"]] = False
            if "preferences" not in self.user_profile:
                self.user_profile["preferences"] = {}
            self.user_profile["preferences"][entities["dislike"]] = False
        
        # Update emotion detection
        self.emotion_detected = self._detect_emotion(user_input)
        
        # Update topic
        new_topic = self._detect_topic(user_input)
        if new_topic and new_topic != self.current_topic:
            if self.current_topic:
                self.topic_history.append(self.current_topic)
            self.current_topic = new_topic
            if "topics_discussed" not in self.user_profile:
                self.user_profile["topics_discussed"] = []
            if new_topic not in self.user_profile["topics_discussed"]:
                self.user_profile["topics_discussed"].append(new_topic)
        
        # Save updated profile
        self._save_user_profile()
    
    def _format_response(self, response_type, **kwargs):
        """Format a response using the appropriate template."""
        if response_type not in self.response_templates:
            response_type = "information"
        
        template = random.choice(self.response_templates[response_type])
        
        # Add default values for any missing keywords
        defaults = {
            "bot_name": self.name,
            "user_name": self.user_name,
            "topic": self.current_topic or "this topic",
            "direct_answer": "Here's what I know:",
            "supporting_details": "",
            "multiple_perspectives": "",
            "knowledge_limitations": "",
            "related_topics": "",
            "preferences": "what you've shared",
            "primary_recommendation": "this option",
            "explanation": "",
            "alt1": "Another option",
            "alt1_explanation": "",
            "alt2": "A second alternative",
            "alt2_explanation": "",
            "alt3": "A third possibility",
            "alt3_explanation": "",
            "problem": "this issue",
            "solution1": "First approach",
            "solution1_details": "",
            "solution1_pros_cons": "",
            "solution2": "Second approach",
            "solution2_details": "",
            "solution2_pros_cons": "",
            "emotion": self.emotion_detected or "this way",
            "situation": "this situation",
            "validation": "That's completely understandable.",
            "perspective": "",
            "support_offer": "I'm here to listen if you want to talk more about it."
        }
        
        # Merge defaults with provided kwargs
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            # Fall back to a simple response if template formatting fails
            return f"I'd like to help you with {kwargs.get('topic', 'this')}. Could you tell me more about what you're looking for?"
    
    def _handle_function_calls(self, user_input):
        """Handle function call requests like time, date, jokes."""
        if re.search(self.intents["time_request"], user_input):
            return self.knowledge_base["functions"]["tell_time"]()
        
        elif re.search(self.intents["date_request"], user_input):
            return self.knowledge_base["functions"]["tell_date"]()
        
        elif re.search(self.intents["joke_request"], user_input):
            return self.knowledge_base["functions"]["tell_joke"]()
        
        return None
    
    def _handle_information_request(self, user_input):
        """Handle requests for factual information."""
        # Simple keyword-based information retrieval
        if "capital of" in user_input.lower():
            country_match = re.search(r"capital of (\w+)", user_input.lower())
            if country_match:
                country = country_match.group(1).upper()
                if country in self.knowledge_base["general_facts"]["capital_cities"]:
                    capital = self.knowledge_base["general_facts"]["capital_cities"][country]
                    return self._format_response("information", 
                        direct_answer=f"The capital of {country} is {capital}.",
                        supporting_details=f"{capital} is the political center and houses the government of {country}.",
                        topic=f"the capital of {country}")
        
        elif "planets" in user_input.lower():
            planets = self.knowledge_base["general_facts"]["planets"]
            return self._format_response("information",
                direct_answer="The planets in our solar system are:",
                supporting_details=", ".join(planets),
                topic="planets in our solar system")
        
        return None
    
    def _handle_recommendation_request(self, user_input):
        """Handle requests for recommendations."""
        if "book" in user_input.lower():
            genre = "fiction"  # Default genre
            for possible_genre in ["fiction", "non_fiction", "science_fiction"]:
                if possible_genre.replace("_", " ") in user_input.lower():
                    genre = possible_genre
                    break
            
            books = self.knowledge_base["recommendations"]["books"][genre]
            primary = books[0]
            alternatives = books[1:4]
            
            return self._format_response("recommendation",
                preferences=f"your interest in {genre.replace('_', ' ')} books",
                primary_recommendation=primary,
                explanation=f"{primary} is a highly acclaimed {genre.replace('_', ' ')} book that many readers enjoy.",
                alt1=alternatives[0],
                alt1_explanation="Another excellent choice in this genre",
                alt2=alternatives[1],
                alt2_explanation="A classic that has stood the test of time",
                alt3=alternatives[2] if len(alternatives) > 2 else "Other works by the same author",
                alt3_explanation="Worth exploring for similar themes")
        
        elif "movie" in user_input.lower() or "film" in user_input.lower():
            genre = "drama"  # Default genre
            for possible_genre in ["action", "drama", "comedy"]:
                if possible_genre in user_input.lower():
                    genre = possible_genre
                    break
            
            movies = self.knowledge_base["recommendations"]["movies"][genre]
            primary = movies[0]
            alternatives = movies[1:] + ["Exploring the director's other works"]
            
            return self._format_response("recommendation",
                preferences=f"your interest in {genre} movies",
                primary_recommendation=primary,
                explanation=f"{primary} is a highly acclaimed {genre} film that many viewers enjoy.",
                alt1=alternatives[0],
                alt1_explanation="Another excellent choice in this genre",
                alt2=alternatives[1] if len(alternatives) > 1 else "Similar themed movies",
                alt2_explanation="Known for similar qualities",
                alt3="Exploring other films in this genre",
                alt3_explanation="To broaden your experience")
        
        return None
    
    def _handle_problem_solving(self, user_input):
        """Handle problem-solving requests."""
        # Simplified problem detection and response
        if "learn" in user_input.lower() and "programming" in user_input.lower():
            return self._format_response("problem_solving",
                problem="wanting to learn programming",
                solution1="Start with an interactive tutorial",
                solution1_details="Websites like Codecademy, freeCodeCamp, or Python.org offer beginner-friendly tutorials",
                solution1_pros_cons="Pros: Immediate feedback and structured learning. Cons: May not cover advanced topics.",
                solution2="Take an online course",
                solution2_details="Platforms like Coursera, edX, or Udemy offer comprehensive programming courses",
                solution2_pros_cons="Pros: Structured curriculum and certification. Cons: May require payment and time commitment.")
        
        elif "budget" in user_input.lower() or "save money" in user_input.lower():
            return self._format_response("problem_solving",
                problem="managing your budget or saving money",
                solution1="Track all expenses",
                solution1_details="Use a spreadsheet or app like Mint or YNAB to track every expense",
                solution1_pros_cons="Pros: Creates awareness of spending patterns. Cons: Requires consistent effort.",
                solution2="Implement the 50/30/20 rule",
                solution2_details="Allocate 50% of income to needs, 30% to wants, and 20% to savings/debt",
                solution2_pros_cons="Pros: Simple framework to follow. Cons: May need adjustment for your specific situation.")
        
        return None
    
    def _handle_emotional_support(self, user_input):
        """Handle requests for emotional support."""
        emotion = self.emotion_detected
        
        if emotion == "sad":
            return self._format_response("emotional_support",
                emotion="sad",
                situation="what you're going through",
                validation="It's natural to feel down sometimes, and your feelings are valid.",
                perspective="While these feelings can be difficult, they're also a normal part of human experience.",
                support_offer="Is there anything specific that's causing you to feel this way?")
        
        elif emotion == "anxious":
            return self._format_response("emotional_support",
                emotion="anxious",
                situation="what's causing your concern",
                validation="Anxiety can be really challenging to deal with, and it's completely understandable.",
                perspective="Sometimes anxiety comes from focusing on things outside our control. It can help to identify what's within your control.",
                support_offer="Would talking through your specific concerns help?")
        
        elif emotion == "confused":
            return self._format_response("emotional_support",
                emotion="confused",
                situation="this topic",
                validation="It's perfectly normal to feel confused when dealing with complex situations.",
                perspective="Confusion often leads to clarity with a bit of time and exploration.",
                support_offer="Would it help if we break this down into smaller pieces?")
        
        return None
    
    def _handle_greeting(self):
        """Handle greeting intents."""
        template = random.choice(self.response_templates["greeting"])
        return template.format(bot_name=self.name)
    
    def _handle_farewell(self):
        """Handle farewell intents."""
        template = random.choice(self.response_templates["farewell"])
        return template
    
    def _handle_unknown_intent(self):
        """Handle unknown intents."""
        responses = [
            f"I'm not quite sure what you're asking. Could you rephrase that?",
            f"I'd like to help, but I'm not sure I understand. Could you explain more?",
            f"I'm still learning, and I'm not sure how to respond to that. Could you try asking in a different way?",
            f"That's an interesting point. Could you elaborate a bit more so I can better assist you?"
        ]
        return random.choice(responses)
    
    def respond(self, user_input):
        """Generate a response to user input."""
        # Add user message to conversation history
        self.conversation_history.append(f"{self.user_name}: {user_input}")
        
        # Detect intent and entities
        intent = self._detect_intent(user_input)
        entities = self._extract_entities(user_input)
        
        # Update conversation context
        self._update_context(user_input, intent, entities)
        
        # Handle different intents
        response = None
        
        # First check for function calls (time, date, jokes)
        response = self._handle_function_calls(user_input)
        
        # Handle specific intents if function call didn't yield a response
        if not response:
            if intent == "greeting":
                response = self._handle_greeting()
            elif intent == "farewell":
                response = self._handle_farewell()
            elif intent == "information_request":
                response = self._handle_information_request(user_input)
            elif intent == "recommendation_request":
                response = self._handle_recommendation_request(user_input)
            elif intent == "problem_statement":
                response = self._handle_problem_solving(user_input)
            elif intent == "emotional_expression":
                response = self._handle_emotional_support(user_input)
            
            # If still no response, handle as unknown intent
            if not response:
                response = self._handle_unknown_intent()
        
        # Add response to conversation history
        self.conversation_history.append(f"{self.name}: {response}")
        
        return response
    
    def start_session(self):
        """Start a new conversation session."""
        self.session_counter += 1
        self.user_profile["session_count"] = self.session_counter
        self.last_interaction_time = datetime.datetime.now()
        self._save_user_profile()
        
        # Determine if this is a returning user
        if self.session_counter > 1:
            return f"Welcome back, {self.user_name}! It's good to see you again. How can I help you today?"
        else:
            return f"Hello! I'm {self.name}, your conversational assistant. What would you like to talk about today?"
    
    def end_session(self):
        """End the current conversation session."""
        self._save_user_profile()
        return f"Thank you for chatting with me today, {self.user_name}. I hope to talk with you again soon!"
    
    def get_conversation_history(self):
        """Return the conversation history."""
        return "\n".join(self.conversation_history)


def main():
    """Main function to run the chatbot."""
    chatbot = ConversationalAIBot()
    
    print("\033[1m" + "=" * 50 + "\033[0m")
    print("\033[1m" + f"Welcome to {chatbot.name} - Conversational AI Chatbot" + "\033[0m")
    print("\033[1m" + "=" * 50 + "\033[0m")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Start the session
    print(f"{chatbot.name}: {chatbot.start_session()}")
    
    while True:
        # Get user input
        user_input = input(f"\n{chatbot.user_name}: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print(f"\n{chatbot.name}: {chatbot.end_session()}")
            break
        
        # Generate and print response
        response = chatbot.respond(user_input)
        print(f"\n{chatbot.name}: {response}")


if __name__ == "__main__":
    main()